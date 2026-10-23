"""serve/test_priority.py - the turn scheduler's priorities: bands, the fair order within a band, reservations, aging
and preemption (serve/turns.py; no process, no GPU), and a preempted request going on through the server.

    python -m unittest serve.test_priority -v
"""
from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.turns import (PRIORITY_MAX, Priorities, Turn, TurnScheduler, priorities_from_config,  # noqa: E402
                         request_priority, request_session, turns_from_config)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def req(key, p=None, cites=(), issued=(), tools=()):
    """A request of session `key` at priority `p`, answering `cites` and (once it runs) making the calls `issued`."""
    return Turn(frozenset(cites), set(issued), set(tools), priority=p, session_key=key)


class PriorityTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.lock = threading.Lock()

    def sched(self, slice_s=0.0, hold_s=0.0, max_wait_s=0.0, **prio):
        prio.setdefault("aging_s", 0.0)
        return TurnScheduler(self.lock, slice_s, max_wait_s, hold_s, ("Agent",), 120.0, clock=self.clock,
                             prio=Priorities(**prio))

    def take(self, s, *tickets):
        """The one of `tickets` that gets the turn now (None: nobody)."""
        for t in tickets:
            if s.acquire(t, 0):
                return t
        return None

    def test_better_band_first(self):
        s = self.sched()
        low, high = s.enter(req("l", 5)), s.enter(req("h", 0))
        self.assertIsNone(self.take(s, low), "5 is a band worse than 0: it waits")
        self.assertIs(self.take(s, high), high)
        s.release(1)
        self.assertIs(self.take(s, low), low)

    def test_fair_share_within_the_band(self):
        """Priorities 0 and 3 (less than a band apart) share the engine about 1.25^3 = 1.95 to 1."""
        s = self.sched()
        runs = {"x": 0, "y": 0}
        for _ in range(60):
            tx, ty = s.enter(req("x", 0)), s.enter(req("y", 3))
            got = self.take(s, tx, ty)
            self.assertIsNotNone(got)
            s.release(1.0)
            s.leave(ty if got is tx else tx)
            runs[got.turn.session_key] += 1
        self.assertAlmostEqual(runs["x"] / runs["y"], 1.95, delta=0.25)

    def test_equal_priorities_take_turns(self):
        s = self.sched()
        got = []
        for _ in range(6):
            ta, tb = s.enter(req("a")), s.enter(req("b"))
            t = self.take(s, ta, tb)
            s.release(1.0)
            s.leave(tb if t is ta else ta)
            got.append(t.turn.session_key)
        self.assertEqual(got, ["a", "b"] * 3)

    def test_new_session_starts_at_the_least_virtual_time(self):
        s = self.sched()
        for _ in range(5):                              # a has used 50 s alone
            t = s.enter(req("a"))
            self.take(s, t)
            s.release(10.0)
        ta, tb = s.enter(req("a")), s.enter(req("b"))
        self.assertEqual(tb.turn.session.vruntime, ta.turn.session.vruntime, "b banks no 50 s of credit")

    def test_session_found_by_its_tool_results(self):
        s = self.sched()
        t = s.enter(Turn(issued={"c1"}, priority=-3))
        self.take(s, t)
        s.release(1)
        follow = s.enter(Turn(frozenset({"c1"})))
        self.assertIs(follow.turn.session, t.turn.session)
        self.assertEqual(s.effective(follow), -3, "the session's priority carries to its follow-up")

    def test_floor_and_range(self):
        s = self.sched(floor=-5)
        self.assertEqual(s.enter(req("a", -20)).turn.session.priority, -5)
        self.assertEqual(s.enter(req("b", 99)).turn.session.priority, PRIORITY_MAX)

    def test_preempt_full_after_min_run(self):
        s = self.sched(min_run_s=2.0)
        low = s.enter(req("l", 5))
        self.take(s, low)
        high = s.enter(req("h", 0))
        self.assertFalse(low.preempt.is_set(), "the running one has had the turn under priority_min_run_s")
        self.clock.t = 2.5
        self.assertFalse(s.acquire(high, 0))
        self.assertTrue(low.preempt.is_set())
        s.release(preempted=True)
        s.requeue(low)
        self.assertIs(self.take(s, low, high), high)
        self.assertEqual(s.view()["priority"]["preemptions"], 1)

    def test_no_preemption_within_the_band(self):
        s = self.sched(min_run_s=0)
        t = s.enter(req("a", 0))
        self.take(s, t)
        s.enter(req("b", -4))
        self.clock.t = 100
        s.acquire(s.waiting[0], 0)
        self.assertFalse(t.preempt.is_set())

    def test_preempt_timeout(self):
        s = self.sched(preempt="timeout", preempt_timeout_s=10)
        low = s.enter(req("l", 5))
        self.take(s, low)
        self.clock.t = 1
        high = s.enter(req("h", 0))
        self.clock.t = 10.5
        s.acquire(high, 0)
        self.assertFalse(low.preempt.is_set(), "the better one has waited 9.5 s")
        self.clock.t = 11.5
        s.acquire(high, 0)
        self.assertTrue(low.preempt.is_set())

    def test_preempt_off_waits_for_the_request(self):
        s = self.sched(preempt="off")
        low = s.enter(req("l", 5))
        self.take(s, low)
        high = s.enter(req("h", 0))
        self.clock.t = 1000
        self.assertFalse(s.acquire(high, 0))
        self.assertFalse(low.preempt.is_set())
        s.release()
        low2 = s.enter(req("l", 5))
        self.assertIs(self.take(s, low2, high), high, "it still goes first at the request's end")

    def test_preempt_not_for_a_one_shot(self):
        s = self.sched(min_run_s=0)
        low = s.enter(req("l", 5), preemptible=False)
        self.take(s, low)
        s.enter(req("h", 0))
        self.assertFalse(low.preempt.is_set())

    def test_reservation_keeps_worse_bands_out(self):
        s = self.sched(reserve_s=60)
        h = s.enter(req("h", 0, issued={"h1"}))
        self.take(s, h)
        s.release(1)                                    # its response made a call: reserved for its follow-up
        low = s.enter(req("l", 5))
        self.clock.t = 30
        self.assertFalse(s.acquire(low, 0), "reserved for h")
        same = s.enter(req("m", 2))
        self.assertIs(self.take(s, same), same, "a request of h's band is not kept out")
        s.release(1)
        self.assertIsNone(s.reserved(), "m's response made no calls: nothing reserved now")
        self.assertIs(self.take(s, low), low)

    def test_reservation_runs_out(self):
        s = self.sched(reserve_s=60)
        h = s.enter(req("h", 0, issued={"h1"}))
        self.take(s, h)
        s.release(1)
        low = s.enter(req("l", 5))
        self.clock.t = 59
        self.assertFalse(s.acquire(low, 0))
        self.clock.t = 61
        self.assertTrue(s.acquire(low, 0))
        self.assertGreater(s.view()["priority"]["reserve_idle_s"], 0)

    def test_follow_up_takes_the_reservation(self):
        s = self.sched(reserve_s=60)
        h = s.enter(req("h", 0, issued={"h1"}))
        self.take(s, h)
        s.release(1)
        low = s.enter(req("l", 5))
        follow = s.enter(Turn(frozenset({"h1"})))
        self.assertIs(self.take(s, low, follow), follow)

    def test_yielding_session_reserves_nothing(self):
        s = self.sched(reserve_s=60)
        for tools, timeout in (({"Agent"}, 0.0), ({"bash"}, 300.0)):
            turn = req("h", 0, issued={"h1"}, tools=tools)
            turn.timeout_s = timeout
            h = s.enter(turn)
            self.take(s, h)
            s.release(1)
            self.assertIsNone(s.reserved(), f"{tools} with a {timeout:g} s timeout: it is yielding")
        h = s.enter(req("h", 0))                        # a response with no calls: its turn is over
        self.take(s, h)
        s.release(1)
        self.assertIsNone(s.reserved())

    def test_preempted_holds_nothing(self):
        s = self.sched(min_run_s=0, reserve_s=60)
        low = s.enter(req("l", 5, issued={"l1"}))       # it had made a call before it was stopped
        self.take(s, low)
        s.enter(req("h", 0))
        s.release(preempted=True)
        self.assertIsNone(s.reserved())
        self.assertFalse(s.holds())

    def test_aging_ends_a_wait(self):
        s = self.sched(reserve_s=10_000, aging_s=30)
        h = s.enter(req("h", 0, issued={"h1"}))
        self.take(s, h)
        s.release(1)
        low = s.enter(req("l", 10))
        self.clock.t = 179
        self.assertFalse(s.acquire(low, 0), "10 - 5 steps = 5: still a band worse")
        self.clock.t = 181
        self.assertTrue(s.acquire(low, 0), "6 steps: 4, in h's band")
        self.assertEqual(low.run_p, 4)

    def test_aged_runner_keeps_its_turn(self):
        """A request that aged into the band is stopped only by one a band better than the priority it ran with."""
        s = self.sched(aging_s=30, min_run_s=0)
        low = s.enter(req("l", 10))
        self.clock.t = 180
        self.take(s, low)                               # it took the turn at 4
        s.enter(req("h", 0))
        self.assertFalse(low.preempt.is_set())
        s.enter(req("u", -1))
        self.assertTrue(low.preempt.is_set())

    def test_hold_within_the_band_only(self):
        s = self.sched(slice_s=30, hold_s=3, reserve_s=0)
        a = s.enter(req("a", 5, issued={"a1"}))
        self.take(s, a)
        s.release(1)
        b = s.enter(req("b", 5))
        self.assertFalse(s.acquire(b, 0), "a's band waits for its follow-up")
        h = s.enter(req("h", 0))
        self.assertIs(self.take(s, h), h, "a better band does not")

    def test_threads(self):
        """A request preempted on a thread gives the turn back; the better one takes it, then the first goes on."""
        s = TurnScheduler(self.lock, 0, 0, prio=Priorities(min_run_s=0, aging_s=0))
        order = []
        low = s.enter(req("l", 5))
        self.assertTrue(s.acquire(low, 1))

        def high():
            t = s.enter(req("h", 0))
            self.assertTrue(s.acquire(t, 5))
            order.append("h")
            s.release()
        th = threading.Thread(target=high)
        th.start()
        self.assertTrue(low.preempt.wait(5))
        s.release(preempted=True)
        s.requeue(low)
        self.assertTrue(s.acquire(low, 5))
        order.append("l")
        s.release()
        th.join(5)
        self.assertEqual(order, ["h", "l"])

    def test_from_config(self):
        lock = threading.Lock()
        s, note = turns_from_config({"turn_slice_s": 30}, lock)
        self.assertEqual(s.prio, Priorities())
        self.assertIn("priority_preempt: full", note)
        s, _ = turns_from_config({"turn_slice_s": 30, "priority": False}, lock)
        self.assertIsNone(s.prio)
        self.assertEqual(turns_from_config({}, lock), (None, None))
        s, note = turns_from_config({"priority": True}, lock)
        self.assertEqual((s.slice_s, s.prio.delta), (0.0, 5))
        self.assertIn("by priority", note)
        p = priorities_from_config({"priority_preempt": "timeout", "priority_preempt_timeout_s": 4,
                                    "priority_preempt_delta": 3, "priority_reserve_s": 0, "priority_aging_s": 0,
                                    "priority_min_run_s": 1, "priority_default": 2, "priority_floor": -10})
        self.assertEqual(p, Priorities(3, "timeout", 4.0, 1.0, 0.0, 0.0, 2, -10))
        self.assertEqual(priorities_from_config({"priority_preempt": False}).preempt, "off")
        s, note = turns_from_config({"turn_slice_s": 30, "parallel": 2}, lock)
        self.assertNotIn("priorit", note)
        for bad in ({"priority": "yes"}, {"priority_preempt": "now"}, {"priority_preempt_delta": 0},
                    {"priority_preempt_delta": 2.5}, {"priority_aging_s": -1}, {"priority_floor": -21},
                    {"priority_default": 20}, {"priority_reserve_s": "60"}):
            with self.assertRaises(ValueError, msg=bad):
                turns_from_config({"turn_slice_s": 30, **bad}, lock)

    def test_request_fields(self):
        self.assertIsNone(request_priority({}))
        self.assertEqual(request_priority({"strata_priority": -3}), -3)
        self.assertEqual(request_priority({"strata_priority": 2.0}), 2)
        self.assertEqual(request_priority({}, {"X-Strata-Priority": " 4 "}), 4)
        self.assertEqual(request_priority({"strata_priority": 1}, {"X-Strata-Priority": "4"}), 1, "the body wins")
        for bad in ({"strata_priority": "high"}, {"strata_priority": True}, {"strata_priority": 1.5}):
            with self.assertRaises(ValueError):
                request_priority(bad)
        with self.assertRaises(ValueError):
            request_priority({}, {"X-Strata-Priority": "x"})
        self.assertIsNone(request_session({}))
        self.assertEqual(request_session({"strata_session": "s1"}), "s1")
        self.assertEqual(request_session({}, {"X-Strata-Session": "s2"}), "s2")
        self.assertEqual(request_session({"metadata": {"user_id": "user_x_session_y"}}), "user_x_session_y")
        with self.assertRaises(ValueError):
            request_session({"strata_session": 5})


class ServicePreemptTest(unittest.TestCase):
    def test_preempted_request_goes_on(self):
        """A low-priority reply is stopped for a high-priority one, then goes on from where it was: its client gets
        the whole reply, and the engine is sent the low prompt, the high one, then the low prompt plus what it wrote.
        This engine has no stop_now: the server leaves the stream (what the engine wrote after is read again)."""
        self.preempt(stop_now=False)

    def test_preempted_with_stop_now(self):
        """An engine that takes a STOP without its stream closing writes a few more tokens: the client gets them,
        and the next pass's prompt holds every one (it continues the engine's live conversation exactly)."""
        seen, wrote = self.preempt(stop_now=True)
        self.assertEqual(seen[2][1], seen[0][1] + wrote[0], "the continuation is the prompt plus all it wrote")

    def preempt(self, stop_now):
        import json
        import time
        import urllib.request
        from serve.frontend import ChatTemplate
        from serve.server import IM_END, ByteTokenizer, Service, serve

        tok = ByteTokenizer()
        long_text = " ".join(f"w{i}" for i in range(80))
        scripts = {"conversation L": "</think>\n" + long_text, "conversation H": "</think>\nurgent"}
        seen, wrote = [], []

        class Conversations:
            """Writes each conversation's script from where its prompt already is (a continuation goes on)."""
            max_context = 1 << 16

            def __init__(self):
                self.base, self.stopping = {}, None

            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                text = tok.decode(ids)
                name = next(k for k in scripts if k in text)
                seen.append((name, len(ids)))
                wrote.append(0)
                self.stopping = None
                base = self.base.setdefault(name, len(ids))
                out = tok.encode(scripts[name], parse_special=True) + tok.encode(IM_END, parse_special=True)
                for t in out[len(ids) - base:][:max_new]:
                    if cancel.is_set():
                        return
                    if self.stopping is not None:
                        self.stopping -= 1
                        if self.stopping < 0:
                            return                      # DONE cancel, three tokens after the STOP
                    time.sleep(0.01)
                    wrote[-1] += 1
                    yield t

        class Stoppable(Conversations):
            def stop_now(self):
                self.stopping = 3
                return True

        svc = Service((Stoppable if stop_now else Conversations)(), tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        svc.turns, _ = turns_from_config({"turn_slice_s": 30, "priority_min_run_s": 0}, svc.fifo)
        httpd = serve(svc, port=0)
        base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
        replies = {}

        def post(name, priority):
            body = json.dumps({"model": "m", "max_tokens": 2000, "strata_priority": priority,
                               "messages": [{"role": "user", "content": name}]}).encode()
            r = urllib.request.Request(base_url + "/v1/chat/completions", data=body,
                                       headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(r, timeout=60) as resp:
                replies[name] = json.loads(resp.read())["choices"][0]["message"]["content"]

        try:
            low = threading.Thread(target=post, args=("conversation L", 5))
            low.start()
            end = time.time() + 10
            while not seen and time.time() < end:
                time.sleep(0.01)
            time.sleep(0.2)                             # L is writing
            high = threading.Thread(target=post, args=("conversation H", 0))
            high.start()
            high.join(30)
            low.join(30)
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertEqual([n for n, _ in seen], ["conversation L", "conversation H", "conversation L"])
        self.assertGreater(seen[2][1], seen[0][1], "the continuation's prompt holds what L had written")
        self.assertEqual(replies["conversation H"].strip(), "urgent")
        self.assertEqual(replies["conversation L"].strip(), long_text, "the client got the whole reply, once")
        self.assertEqual(svc.turns.view()["priority"]["preemptions"], 1)
        return seen, wrote


if __name__ == "__main__":
    unittest.main()
