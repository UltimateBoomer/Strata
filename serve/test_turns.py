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
from serve.turns import MIN_SHARED, TurnPolicy, TurnScheduler, shared_prefix, turns_from_config  # noqa: E402

A = list(range(1, 3001))              # conversation A's prompt
B = list(range(100001, 103001))       # conversation B's
C = list(range(200001, 203001))


def more(p, n=200, tag=0):
    """The next turn of the same conversation: its history plus n new tokens."""
    return p + [900000 + tag * 1000 + i for i in range(n)]


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class TurnsTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.lock = threading.Lock()

    def run_one(self, s, ids, used):
        """One request: enter, take the turn at once (it must be its turn), use `used` tokens, release."""
        t = s.enter(ids)
        self.assertTrue(s.acquire(t, 0))
        s.release(used)

    def order(self, s, prompts, used=500):
        """Every prompt waiting at once, in this arrival order: the order they get the turn in."""
        tickets = [s.enter(p) for p in prompts]
        got = []
        while tickets:
            for t in tickets:
                if s.acquire(t, 0):
                    got.append(prompts.index(t.ids))
                    tickets.remove(t)
                    s.release(used)
                    break
            else:
                self.fail("nobody's turn")
        return got

    def test_shared_prefix_and_continues(self):
        self.assertEqual(shared_prefix([1, 2, 3], [1, 2, 4]), 2)
        s = TurnScheduler(self.lock, 4096, clock=self.clock)
        self.assertFalse(s.continues(A))                # nothing ran yet
        self.run_one(s, A, 100)
        self.assertTrue(s.continues(more(A)))
        self.assertTrue(s.continues(A[:2000] + [7] * 900), "the last reply re-rendered: most of the history shared")
        self.assertFalse(s.continues(A[:1000] + B), "too little shared")
        self.assertFalse(s.continues(B))
        short = TurnScheduler(self.lock, 4096, clock=self.clock)
        self.run_one(short, list(range(MIN_SHARED - 1)), 1)
        self.assertFalse(short.continues(list(range(MIN_SHARED - 1)) + [5]), "a short history never counts")

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


class ConfigTest(unittest.TestCase):
    def test_from_config(self):
        lock = threading.Lock()
        self.assertEqual(turns_from_config({}, lock), (None, None))
        self.assertEqual(turns_from_config({"turn_slice_s": 0}, lock), (None, None))
        s, note = turns_from_config({"turn_slice_s": 30, "turn_max_wait_s": 90}, lock)
        self.assertIsInstance(s, TurnScheduler)
        self.assertEqual((s.slice_s, s.max_wait_s, s.lock), (30.0, 90.0, lock))
        self.assertIn("30 s", note)
        s, note = turns_from_config({"turn_slice_s": 30, "parallel": 2}, lock)
        self.assertIsInstance(s, TurnPolicy)
        self.assertIn("batch slot", note)
        for bad in ({"turn_slice_s": "30s"}, {"turn_slice_s": -1}, {"turn_slice_s": True},
                    {"turn_slice_s": 1, "turn_max_wait_s": "x"}):
            with self.assertRaises(ValueError):
                turns_from_config(bad, lock)


class ServiceTest(unittest.TestCase):
    """Through the HTTP server: a conversation's next turn that arrives after another client's request still goes
    first while its budget lasts."""

    def test_order_through_the_server(self):
        import json
        import urllib.request
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, Service, serve

        gate, seen = threading.Event(), []

        class Gated(MockEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                seen.append(list(ids))
                if len(seen) == 1:
                    gate.wait(10)                       # the first request holds the turn until the others queue
                yield from super().generate(ids, max_new, sampling, cancel, embeddings)

        tok = ByteTokenizer()
        engine = Gated(tok, "</think>\nok", max_context=1 << 16)
        svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        svc.turns, _ = turns_from_config({"turn_slice_s": 600}, svc.fifo)
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        c1 = [{"role": "user", "content": "conversation C " + "c" * 3000}]
        c2 = c1 + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "and then"}]
        b = [{"role": "user", "content": "conversation B " + "b" * 3000}]

        def post(messages):
            body = json.dumps({"model": "m", "max_tokens": 8, "messages": messages}).encode()
            req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                r.read()

        def wait_for(cond):
            for _ in range(500):
                if cond():
                    return
                threading.Event().wait(0.01)
            self.fail("timed out")

        try:
            threads = [threading.Thread(target=post, args=(c1,))]
            threads[0].start()
            wait_for(lambda: len(seen) == 1)
            for m, n in ((b, 1), (c2, 2)):              # B arrives, then C's next turn
                threads.append(threading.Thread(target=post, args=(m,)))
                threads[-1].start()
                wait_for(lambda: svc.turns.view()["waiting"] == n)
            gate.set()
            for t in threads:
                t.join(30)
        finally:
            httpd.shutdown()
            httpd.server_close()
        text = [tok.decode(p) for p in seen]
        self.assertEqual(len(text), 3)
        self.assertIn("conversation C", text[1], "C's next turn went before B")
        self.assertIn("and then", text[1])
        self.assertIn("conversation B", text[2])
        self.assertEqual((svc.turns.view()["kept"], svc.turns.view()["switches"]), (1, 1))


class BatchTest(unittest.TestCase):
    """With "parallel": the engine's admissions (StrataEngine._admit_next / _take_control) take the policy, the lanes
    being the batch slots' held conversations."""

    def setUp(self):
        from serve.test_admit_order import engine_without_process
        self.e = engine_without_process(2)
        self.e.slot_held = [[], []]
        self.e.slot_turn_used = [0.0, 0.0]
        self.e.turn_policy = TurnPolicy(30, 120)

    def entry(self, ids, at):
        """A waiting admission `at` seconds after a moment just now (the engine compares with time.monotonic())."""
        import time as t
        return [len(ids), t.monotonic() - 10 + at, True, ids]

    def test_pick_with_slots(self):
        e = self.e
        a, b = self.entry(B, 1.0), self.entry(more(A), 2.0)
        e.wait_lens = [a, b]
        self.assertIs(e._admit_next(), a, "an empty slot is free: B pushes nobody out, arrival order")
        e.slot_held = [A, C]                            # both slots hold a conversation
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
        e.slot_held = [A, C]
        order, cancels = [], []
        self.assertTrue(e.ctl.acquire(timeout=1))       # a read in progress
        def go(name, ids):
            c = threading.Event()
            cancels.append(c)
            gen = e._take_control(c, len(ids), new=True, ids=ids)
            try:
                while True:
                    next(gen)
            except StopIteration as stop:
                if stop.value:
                    order.append(name)
                    t.sleep(0.05)
                    e.ctl.release()
        for n, (name, ids) in enumerate((("B", B), ("A2", more(A))), 1):
            threading.Thread(target=go, args=(name, ids), daemon=True).start()
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
