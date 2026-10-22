"""serve/test_turns.py - "turn_slice_s": the conversation in the engine keeps its lane (the single turn, or a batch
slot) for a slice of engine time before another waiting one takes it (serve/turns.py; no process, no GPU).

    python -m unittest serve.test_turns -v
"""
from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.turns import (HOLD_SKIP_TOOLS, Turn, TurnPolicy, TurnScheduler, call_timeout_s,  # noqa: E402
                         cited_calls, timeout_units, turns_from_config)

def step(name):
    """A response that made one call, `name`."""
    return Turn(issued={name})


def more(t, tag=0):
    """The next step of `t`'s response: its tool results answer t's calls (and it makes one of its own)."""
    return Turn(frozenset(t.issued), {f"{sorted(t.issued)[0]}.{tag}"})


A, B, C = step("a"), step("b"), step("c")     # three conversations' responses


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class TurnsTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.lock = threading.Lock()

    def run_one(self, s, turn, used):
        """One request: enter, take the turn at once (it must be its turn), use `used` seconds, release."""
        t = s.enter(turn)
        self.assertTrue(s.acquire(t, 0))
        s.release(used)

    def order(self, s, prompts, used=500):
        """Every prompt waiting at once, in this arrival order: the order they get the turn in."""
        tickets = [s.enter(p) for p in prompts]
        got = []
        while tickets:
            for t in tickets:
                if s.acquire(t, 0):
                    got.append(prompts.index(t.turn))
                    tickets.remove(t)
                    s.release(used)
                    break
            else:
                self.fail("nobody's turn")
        return got

    def test_continues(self):
        s = TurnScheduler(self.lock, 4096, clock=self.clock)
        self.assertFalse(s.continues(A))                # nothing ran yet
        self.run_one(s, A, 100)
        self.assertTrue(s.continues(more(A)), "its tool results answer A's call")
        self.assertTrue(s.continues(Turn(frozenset({"a", "x"}))), "one of the calls answered is enough")
        self.assertTrue(s.continues(A), "the same response (a later pass)")
        self.assertFalse(s.continues(Turn()), "a new user message: no results, a turn of its own")
        self.assertFalse(s.continues(B))
        self.assertFalse(s.continues(more(B)), "answers another response's call")
        self.assertFalse(TurnPolicy.continues(more(A), None))

    def test_stays_on_the_conversation_within_the_budget(self):
        s = TurnScheduler(self.lock, 4096, clock=self.clock)
        self.run_one(s, A, 500)
        # B arrived first, then A's next turn: A goes first while its budget lasts
        self.assertEqual(self.order(s, [B, more(A)], used=500), [1, 0])
        self.assertEqual(s.view()["kept"], 1)

    def test_switches_once_the_budget_is_used(self):
        s = TurnScheduler(self.lock, 1000, clock=self.clock)
        self.run_one(s, A, 1200)                        # A used more than its budget
        self.assertEqual(self.order(s, [B, more(A)], used=100), [0, 1], "the oldest goes: B")
        self.assertEqual(s.view()["switches"], 2)       # A -> B -> A

    def test_budget_counts_several_requests(self):
        s = TurnScheduler(self.lock, 1000, clock=self.clock)
        a = A
        self.run_one(s, a, 400)
        for i in range(2):                              # two more of A's turns jump B: 400 + 400 + 400
            b = s.enter(B)
            a = more(a, tag=i)
            ta = s.enter(a)
            self.assertFalse(s.acquire(b, 0), "B waits while A is within its budget")
            self.assertTrue(s.acquire(ta, 0))
            s.release(400)
            s.leave(b)
        b, ta = s.enter(B), s.enter(more(a, tag=9))
        self.assertTrue(s.acquire(b, 0), "past the budget B goes")
        s.release(10)
        s.leave(ta)

    def test_waiting_too_long_wins(self):
        s = TurnScheduler(self.lock, 100000, max_wait_s=60, clock=self.clock)
        self.run_one(s, A, 10)
        b = s.enter(B)
        self.clock.t = 61                               # B has waited past turn_max_wait_s
        a = s.enter(more(A))
        self.assertFalse(s.acquire(a, 0))
        self.assertTrue(s.acquire(b, 0))
        s.release(1)
        self.assertTrue(s.acquire(a, 0))
        s.release(1)

    def test_no_budget_is_arrival_order(self):
        s = TurnScheduler(self.lock, 0, clock=self.clock)
        self.run_one(s, A, 10)
        self.assertEqual(self.order(s, [B, C, more(A)]), [0, 1, 2])

    def test_leave_and_outside_holders(self):
        s = TurnScheduler(self.lock, 4096, clock=self.clock)
        self.run_one(s, A, 10)
        b = s.enter(B)
        a = s.enter(more(A))
        s.leave(a)                                      # A's client left while waiting: B's turn now
        self.lock.acquire()                             # an image encode holds the engine
        self.assertFalse(s.acquire(b, 0), "not while something else holds the lock")
        self.lock.release()
        self.assertTrue(s.acquire(b, 0))
        s.release(5)
        self.assertEqual(s.view()["waiting"], 0)

    def test_threads_wait_for_their_turn(self):
        s = TurnScheduler(self.lock, 4096, clock=self.clock)
        self.run_one(s, A, 10)
        hold = s.enter(C)
        self.assertTrue(s.acquire(hold, 0))           # C runs; B then A's next turn arrive
        got = []
        def req(p, name):
            t = s.enter(p)
            while not s.acquire(t, 0.2):
                pass
            got.append(name)
            s.release(10)
        tb = threading.Thread(target=req, args=(B, "B"))
        tb.start()
        while s.view()["waiting"] < 1:
            pass
        ta = threading.Thread(target=req, args=(more(C), "C2"))
        ta.start()
        while s.view()["waiting"] < 2:
            pass
        s.release(10)                                 # C's turn ends: its next turn goes before B
        tb.join(5)
        ta.join(5)
        self.assertEqual(got, ["C2", "B"])


class HoldTest(unittest.TestCase):
    """"turn_hold_s": after a response that made tool calls, the turn waits a while for its follow-up."""

    def setUp(self):
        self.clock = Clock()
        self.lock = threading.Lock()

    def sched(self, hold_s=3, max_wait_s=120):
        return TurnScheduler(self.lock, 600, max_wait_s=max_wait_s, hold_s=hold_s, skip_tools=("Agent",),
                             skip_timeout_s=120, clock=self.clock)

    def run_one(self, s, turn):
        t = s.enter(turn)
        self.assertTrue(s.acquire(t, 0))
        s.release(1)

    def test_hold_hit_then_miss(self):
        s = self.sched()
        self.run_one(s, A)                              # A called a tool and gave the turn back at 0
        b = s.enter(B)
        self.assertTrue(s.view()["holding"])
        self.assertFalse(s.acquire(b, 0), "held for A's follow-up")
        self.clock.t = 1.0
        a2 = s.enter(more(A))                           # the tool came back after 1 s
        self.assertTrue(s.acquire(a2, 0))
        s.release(1)                                    # A's next step called a tool too: held again from 1
        self.clock.t = 3.9
        self.assertFalse(s.acquire(b, 0))
        self.clock.t = 4.0                              # 3 s: the hold ran out, B goes
        self.assertTrue(s.acquire(b, 0))
        s.release(1)
        v = s.view()
        self.assertEqual((v["hold_hits"], v["hold_misses"], v["kept"], v["switches"]), (1, 1, 1, 1))
        self.assertAlmostEqual(v["hold_idle_s"], 1.0 + 3.0)

    def test_alone_the_hold_costs_nothing(self):
        s = self.sched()
        self.run_one(s, A)
        self.clock.t = 0.5
        self.run_one(s, more(A))                        # nobody else waited: no hit, no idle time counted
        v = s.view()
        self.assertEqual((v["hold_hits"], v["hold_misses"], v["hold_idle_s"]), (0, 0, 0.0))

    def test_no_hold(self):
        cases = {"no calls": Turn(),
                 "a skipped tool": Turn(issued={"x"}, tools={"Read", "Agent"}),
                 "a long timeout": Turn(issued={"x"}, tools={"Bash"}, timeout_s=300)}
        for why, turn in cases.items():
            s = self.sched()
            self.run_one(s, turn)
            b = s.enter(B)
            self.assertTrue(s.acquire(b, 0), why)
            s.release(1)
        s = self.sched()
        self.run_one(s, Turn(issued={"x"}, tools={"Bash"}, timeout_s=60))
        self.assertTrue(s.view()["holding"], "a timeout within turn_hold_skip_timeout_s")
        s = self.sched(hold_s=0)
        self.run_one(s, A)
        self.assertTrue(s.acquire(s.enter(B), 0), "turn_hold_s 0")
        s.release(1)
        s = self.sched()
        t = s.enter(A)
        self.assertTrue(s.acquire(t, 0))
        s.release(601)                                  # past its slice
        self.assertTrue(s.acquire(s.enter(B), 0), "past the slice")

    def test_max_wait_beats_the_hold(self):
        s = self.sched(max_wait_s=60)
        t = s.enter(A)
        self.assertTrue(s.acquire(t, 0))                # A runs from 0
        b = s.enter(B)                                  # B waits from 0
        self.clock.t = 60.0
        s.release(1)                                    # A made a call: held, but B has waited 60 s
        self.assertTrue(s.acquire(b, 0))
        s.release(1)

    def test_calls_counted_after_the_release(self):
        s = self.sched()
        late = Turn()
        self.run_one(s, late)
        late.issued.add("late")                         # a call the parser completed as the response ended
        self.assertFalse(s.acquire(s.enter(B), 0), "held once the call is known")
        self.assertTrue(s.acquire(s.enter(Turn(frozenset({"late"}))), 0))
        s.release(1)

    def test_call_timeout(self):
        self.assertEqual(call_timeout_s({"command": "make", "timeout": 600000}), 600.0, "no schema: >= 1000 is ms")
        self.assertEqual(call_timeout_s({"command": "make", "timeout": 300}), 300.0, "no schema: below 1000 is s")
        self.assertEqual(call_timeout_s({"timeout_ms": "1500"}), 1.5)
        self.assertEqual(call_timeout_s({"command": "ls"}), 0.0)
        self.assertEqual(call_timeout_s({"timeout": True}), 0.0)
        self.assertEqual(call_timeout_s("not a dict"), 0.0)
        self.assertEqual(call_timeout_s({"command": "sleep 900", "timeout": 600000, "run_in_background": True}), 0.0,
                         "in the background it returns at once")
        self.assertEqual(call_timeout_s({"command": "x", "timeout": 900, "background": True}), 0.0)

    def test_timeout_units_from_the_schemas(self):
        tools = [
            # Claude Code
            {"name": "Bash", "description": "Executes a given bash command...", "parameters": {"type": "object",
             "properties": {"command": {"type": "string"},
                            "timeout": {"type": "number", "description": "Optional timeout in milliseconds (max 600000)"},
                            "run_in_background": {"type": "boolean"}}}},
            # pi
            {"name": "bash", "description": "Execute a bash command in the current working directory. ... Optionally "
             "provide a timeout in seconds.", "parameters": {"type": "object", "properties": {
                 "command": {"type": "string", "description": "Shell command to execute"},
                 "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"}}}},
            # Hermes Agent
            {"name": "terminal", "description": "Execute shell commands...", "parameters": {"type": "object",
             "properties": {"command": {"type": "string"}, "background": {"type": "boolean"},
                            "timeout": {"type": "integer", "minimum": 1, "description":
                                        "Max seconds (default 180, foreground max 600). Finishing early returns at once, "
                                        "so set it generously."}}}},
            {"name": "shell", "parameters": {"type": "object", "properties": {"timeout_ms": {"type": "number"}}}},
            {"name": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
            "not a tool",
        ]
        units = timeout_units(tools)
        self.assertEqual(units, {"Bash": ("timeout", 0.001), "bash": ("timeout", 1.0), "terminal": ("timeout", 1.0),
                                 "shell": ("timeout_ms", 0.001)})
        self.assertEqual(call_timeout_s({"command": "make", "timeout": 300000}, units["Bash"]), 300.0)
        self.assertEqual(call_timeout_s({"command": "make", "timeout": 120}, units["bash"]), 120.0)
        self.assertEqual(call_timeout_s({"command": "make", "timeout": 1800}, units["terminal"]), 1800.0,
                         "seconds as the schema says, though 1000 or more")
        self.assertEqual(call_timeout_s({"command": "ls"}, units["bash"]), 0.0)

    def test_skip_tools_cover_the_agents(self):
        for name in ("Agent", "Task", "AskUserQuestion", "delegate_task", "clarify"):
            self.assertIn(name, HOLD_SKIP_TOOLS)
        for name in ("Bash", "bash", "terminal", "web_search", "read_file"):
            self.assertNotIn(name, HOLD_SKIP_TOOLS)


class ConfigTest(unittest.TestCase):
    def test_from_config(self):
        lock = threading.Lock()
        self.assertEqual(turns_from_config({}, lock), (None, None))
        self.assertEqual(turns_from_config({"turn_slice_s": 0}, lock), (None, None))
        s, note = turns_from_config({"turn_slice_s": 30, "turn_max_wait_s": 90}, lock)
        self.assertIsInstance(s, TurnScheduler)
        self.assertEqual((s.slice_s, s.max_wait_s, s.lock), (30.0, 90.0, lock))
        self.assertEqual((s.hold_s, s.skip_tools, s.skip_timeout_s), (3.0, frozenset(HOLD_SKIP_TOOLS), 120.0))
        self.assertIn("30 s", note)
        self.assertIn("up to 3 s", note)
        s, note = turns_from_config({"turn_slice_s": 30, "turn_hold_s": 0, "turn_hold_skip_tools": ["X"],
                                     "turn_hold_skip_timeout_s": 0}, lock)
        self.assertEqual((s.hold_s, s.skip_tools, s.skip_timeout_s), (0.0, frozenset({"X"}), 0.0))
        self.assertNotIn("turn_hold_s", note)
        s, note = turns_from_config({"turn_slice_s": 30, "parallel": 2}, lock)
        self.assertIsInstance(s, TurnPolicy)
        self.assertIn("batch slot", note)
        for bad in ({"turn_slice_s": "30s"}, {"turn_slice_s": -1}, {"turn_slice_s": True},
                    {"turn_slice_s": 1, "turn_max_wait_s": "x"}, {"turn_slice_s": 1, "turn_hold_s": -1},
                    {"turn_slice_s": 1, "turn_hold_skip_tools": "Agent"}):
            with self.assertRaises(ValueError):
                turns_from_config(bad, lock)


class CitedCallsTest(unittest.TestCase):
    """The call IDs a request's tool results answer, in each API's form."""

    def test_chat_completions(self):
        req = {"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "tool_calls": [{"id": "call_old", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_old", "content": "1"},
            {"role": "assistant", "tool_calls": [{"id": "call_1"}, {"id": "call_2"}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "a"},
            {"role": "tool", "tool_call_id": "call_2", "content": "b"},
            {"role": "user", "content": "a note added after the results"}]}
        self.assertEqual(cited_calls("/v1/chat/completions", req), {"call_1", "call_2"})
        req["messages"].append({"role": "assistant", "content": "done"})
        req["messages"].append({"role": "user", "content": "next question"})
        self.assertEqual(cited_calls("/v1/chat/completions", req), frozenset(), "a new user turn answers nothing")
        self.assertEqual(cited_calls("/v1/chat/completions", {"messages": "not a list"}), frozenset())

    def test_anthropic_messages(self):
        req = {"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "f", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "a"},
                                         {"type": "text", "text": "<system-reminder>...</system-reminder>"}]}]}
        self.assertEqual(cited_calls("/v1/messages", req), {"call_1"})
        req["messages"][-1] = {"role": "user", "content": "a plain user message"}
        self.assertEqual(cited_calls("/v1/messages", req), frozenset())

    def test_responses(self):
        req = {"input": [
            {"type": "message", "role": "user", "content": "go"},
            {"type": "function_call", "call_id": "call_old", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_old", "output": "1"},
            {"type": "reasoning", "summary": []},
            {"type": "function_call", "call_id": "call_1", "name": "f", "arguments": "{}"},
            {"type": "custom_tool_call", "call_id": "call_2", "name": "g", "input": ""},
            {"type": "function_call_output", "call_id": "call_1", "output": "a"},
            {"type": "custom_tool_call_output", "call_id": "call_2", "output": "b"}]}
        self.assertEqual(cited_calls("/v1/responses", req), {"call_1", "call_2"})
        self.assertEqual(cited_calls("/v1/responses", {"input": "a string"}), frozenset())


class ServiceTest(unittest.TestCase):
    """Through the HTTP server: the next step of the response that ran last goes before another client's request that
    arrived earlier, matched by the ID of the call it answers."""

    def test_order_through_the_server(self):
        import json
        import urllib.request
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, Service, serve

        seen = []

        class Recording(MockEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                seen.append(list(ids))
                yield from super().generate(ids, max_new, sampling, cancel, embeddings)

        tok = ByteTokenizer()
        call = "</think>\n\n<tool_call>\n<function=look>\n<parameter=q>\nx\n</parameter>\n</function>\n</tool_call>"
        engine = Recording(tok, [call, "</think>\nok"], max_context=1 << 16)
        svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        svc.turns, _ = turns_from_config({"turn_slice_s": 600}, svc.fifo)
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        tools = [{"type": "function", "function": {"name": "look", "parameters": {
            "type": "object", "properties": {"q": {"type": "string"}}}}}]
        c1 = [{"role": "user", "content": "conversation C"}]
        b = [{"role": "user", "content": "conversation B"}]

        def post(messages):
            body = json.dumps({"model": "m", "max_tokens": 200, "messages": messages, "tools": tools}).encode()
            req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())

        def wait_for(cond):
            for _ in range(500):
                if cond():
                    return
                threading.Event().wait(0.01)
            self.fail("timed out")

        try:
            msg = post(c1)["choices"][0]["message"]       # C's response calls a tool; it holds the lane
            cid = msg["tool_calls"][0]["id"]
            c2 = c1 + [msg, {"role": "tool", "tool_call_id": cid, "content": "found"}]
            svc.fifo.acquire()                           # something outside the turns holds the engine meanwhile
            threads = []
            for m, n in ((b, 1), (c2, 2)):              # B arrives, then the next step of C's response
                threads.append(threading.Thread(target=post, args=(m,)))
                threads[-1].start()
                wait_for(lambda: svc.turns.view()["waiting"] == n)
            svc.fifo.release()
            for t in threads:
                t.join(30)
        finally:
            httpd.shutdown()
            httpd.server_close()
        text = [tok.decode(p) for p in seen]
        self.assertEqual(len(text), 3)
        self.assertIn("found", text[1], "the next step of C's response went before B")
        self.assertIn("conversation B", text[2])
        self.assertEqual((svc.turns.view()["kept"], svc.turns.view()["switches"]), (1, 1))


    def test_a_long_timeout_is_not_held_for(self):
        """pi's bash, its timeout in seconds as its schema says: 300 s is above turn_hold_skip_timeout_s."""
        import json
        import urllib.request
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, Service, serve

        tok = ByteTokenizer()
        call = ("</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\nmake\n</parameter>\n"
                "<parameter=timeout>\n300\n</parameter>\n</function>\n</tool_call>")
        svc = Service(MockEngine(tok, call, max_context=1 << 16), tok,
                      ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        svc.turns, _ = turns_from_config({"turn_slice_s": 600}, svc.fifo)
        httpd = serve(svc, port=0)
        tools = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"}}}}}]
        body = json.dumps({"model": "m", "max_tokens": 200, "tools": tools,
                           "messages": [{"role": "user", "content": "build it"}]}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                msg = json.loads(r.read())["choices"][0]["message"]
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "bash")
        lane = svc.turns.lane
        self.assertEqual((lane.tools, lane.timeout_s), ({"bash"}, 300.0))
        self.assertEqual(lane.issued, {msg["tool_calls"][0]["id"]})
        self.assertFalse(svc.turns.view()["holding"], "a 300 s timeout: not held for")


class BatchTest(unittest.TestCase):
    """With "parallel": the engine's admissions (StrataEngine._admit_next / _take_control) take the policy, the lanes
    being the batch slots' held conversations."""

    def setUp(self):
        from serve.test_admit_order import engine_without_process
        self.e = engine_without_process(2)
        self.e.slot_held = [[], []]
        self.e.slot_turn_used = [0.0, 0.0]
        self.e.slot_turn = [None, None]
        self.e.turn_policy = TurnPolicy(30, 120)

    def entry(self, turn, at):
        """A waiting admission `at` seconds after a moment just now (the engine compares with time.monotonic())."""
        import time as t
        return [3000, t.monotonic() - 10 + at, True, turn]

    def test_pick_with_slots(self):
        e = self.e
        a, b = self.entry(B, 1.0), self.entry(more(A), 2.0)
        e.wait_lens = [a, b]
        self.assertIs(e._admit_next(), a, "an empty slot is free: B pushes nobody out, arrival order")
        e.slot_held, e.slot_turn = [[1], [2]], [A, C]   # both slots hold a conversation
        self.assertIs(e._admit_next(), b, "B would push one out: A's next turn first while A is within its slice")
        e.slot_turn_used = [31.0, 0.0]
        self.assertIs(e._admit_next(), a, "A is past its slice: arrival order")
        e.slot_turn_used = [0.0, 0.0]
        e.wait_lens = [self.entry(more(C), 1.0), b]
        self.assertIs(e._admit_next(), e.wait_lens[0], "the oldest continues a slot: it goes")

    def test_take_control_order(self):
        """Through the control lines with threads: B arrives first, A's next turn second; both slots are held."""
        import time as t
        e = self.e
        e.slot_held, e.slot_turn = [[1], [2]], [A, C]
        order, cancels = [], []
        self.assertTrue(e.ctl.acquire(timeout=1))       # a read in progress
        def go(name, turn):
            c = threading.Event()
            cancels.append(c)
            gen = e._take_control(c, 3000, new=True, turn=turn)
            try:
                while True:
                    next(gen)
            except StopIteration as stop:
                if stop.value:
                    order.append(name)
                    t.sleep(0.05)
                    e.ctl.release()
        for n, (name, turn) in enumerate((("B", B), ("A2", more(A))), 1):
            threading.Thread(target=go, args=(name, turn), daemon=True).start()
            end = t.time() + 5
            while len(e.wait_lens) < n and t.time() < end:
                t.sleep(0.01)
            t.sleep(0.02)
        e.ctl.release()
        end = t.time() + 10
        while len(order) < 2 and t.time() < end:
            t.sleep(0.02)
        for c in cancels:
            c.set()
        self.assertEqual(order, ["A2", "B"])


if __name__ == "__main__":
    unittest.main()
