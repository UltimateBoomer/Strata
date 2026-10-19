"""serve/turns.py - which waiting request the engine takes next, with a time slice per conversation (the config's
"turn_slice_s", opt-in).

Every switch to another conversation has a cost: one request at a time, the engine parks the outgoing conversation and
reads the incoming one back (with --conversation-cache-dir: a file written and a file read, hundreds of MB each); with
"parallel" (batch slots), a new conversation in a slot pushes out the one that slot held, whose next turn then reads
its history again.  When several clients take turns - an agent and its subagents, two harnesses - their requests
interleave and nearly every request is such a switch.

The conversations the engine holds are its lanes: the single session, or each batch slot's.  A lane keeps its place
while it has used less than the slice of engine time since it took the lane (each request's time on the engine, from
taking the turn or slot to giving it back): a waiting request that continues a lane within its slice (its prompt shares
most of what the lane holds) goes before one that would push a conversation out.  A request that pushes nobody out (it
continues a lane, or an empty lane is free) goes in arrival order; so does everything once the lanes it would wait for
are past their slice; and one that has waited "turn_max_wait_s" goes next whatever the slices say, so nobody starves.
A turn changes hands between requests only: a request is never stopped for the slice.  Without a slice: arrival order.

Time, not tokens: what the other clients wait is time, and a switch costs about a fixed time (a park and a read-back),
so a slice of T seconds bounds the switching overhead to that cost over T.  A token count would weigh a prompt read
(~1-2K tokens/s) and a generated token (~30/s) alike.
"""
from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field

# a waiting prompt continues a lane when they share at least this much of what the lane holds (a client re-sends its
# history with the last reply re-rendered, so the tail differs) and at least MIN_SHARED tokens
SHARE = 0.5
MIN_SHARED = 1024


def shared_prefix(a, b) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class TurnPolicy:
    """The order alone, over any number of lanes: pick() is asked with the waiting requests and the lanes as they are."""

    def __init__(self, slice_s: float = 0.0, max_wait_s: float = 120.0, clock=time.monotonic):
        self.slice_s = max(0.0, float(slice_s or 0))
        self.max_wait_s = max(0.0, float(max_wait_s or 0))
        self.clock = clock

    @staticmethod
    def continues(ids, held) -> bool:
        """`ids` is the next turn of the conversation that `held` holds: it shares most of it, and at least
        MIN_SHARED tokens."""
        if not held:
            return False
        n = shared_prefix(ids, held)
        return n >= MIN_SHARED and n >= SHARE * len(held)

    def pick(self, waiting, lanes, room: bool):
        """The waiting request that goes next.  `waiting`: (key, ids, arrived) in arrival order; `lanes`: (held ids,
        seconds used) of each conversation the engine holds; `room`: an empty lane is free (a new conversation pushes
        nobody out).  None when nobody waits."""
        if not waiting:
            return None
        key, ids, arrived = waiting[0]
        if self.slice_s <= 0 or room or (self.max_wait_s > 0 and self.clock() - arrived >= self.max_wait_s):
            return key
        if any(self.continues(ids, held) for held, _ in lanes):
            return key                                  # the oldest pushes nobody out
        for k, w_ids, _ in waiting:                     # the oldest that continues a lane within its slice
            if any(used < self.slice_s and self.continues(w_ids, held) for held, used in lanes):
                return k
        return key


@dataclass
class Ticket:
    ids: list
    arrived: float = field(default_factory=time.monotonic)
    seq: int = 0


class TurnScheduler:
    """One request at a time: orders the requests that wait for the engine's single turn (`lock`, which other work -
    an image encode, a VRAM resize - may also take directly).  One lane, the engine's session.  acquire() returns True
    once the ticket holds `lock`; release() gives it back and charges the time it was held."""

    def __init__(self, lock, slice_s: float = 0.0, max_wait_s: float = 120.0, clock=time.monotonic):
        self.lock = lock
        self.policy = TurnPolicy(slice_s, max_wait_s, clock)
        self.slice_s, self.max_wait_s, self.clock = self.policy.slice_s, self.policy.max_wait_s, clock
        self.cv = threading.Condition()
        self.waiting: list[Ticket] = []
        self.seq = itertools.count()
        self.last_ids: list | None = None   # the prompt of the request that ran last
        self.used = 0.0                     # seconds the current conversation has held the turn since it took it
        self.since: float | None = None     # when the current holder took the turn
        self.switches = 0                   # turns that went to another conversation
        self.kept = 0                       # turns that went to a later arrival of the current conversation

    def continues(self, ids) -> bool:
        return TurnPolicy.continues(ids, self.last_ids)

    def enter(self, ids) -> Ticket:
        t = Ticket(list(ids), self.clock(), next(self.seq))
        with self.cv:
            self.waiting.append(t)
        return t

    def leave(self, t: Ticket):
        """A ticket that gave up waiting (its client left)."""
        with self.cv:
            if t in self.waiting:
                self.waiting.remove(t)
                self.cv.notify_all()

    def next_ticket(self) -> Ticket | None:
        """The ticket whose turn it is (the caller holds self.cv)."""
        order = sorted(self.waiting, key=lambda t: t.seq)
        lanes = [(self.last_ids, self.used)] if self.last_ids else []
        return self.policy.pick([(t, t.ids, t.arrived) for t in order], lanes, room=not lanes)

    def acquire(self, t: Ticket, timeout: float) -> bool:
        """Waits up to `timeout` seconds for `t`'s turn and the lock; False on timeout (the caller beats and retries)."""
        end = self.clock() + timeout
        with self.cv:
            while True:
                left = end - self.clock()
                if self.next_ticket() is t and self.lock.acquire(blocking=False):
                    self.waiting.remove(t)
                    if self.continues(t.ids):
                        if self.waiting and t.seq > min(x.seq for x in self.waiting):
                            self.kept += 1
                    else:
                        self.used = 0.0
                        if self.last_ids is not None:
                            self.switches += 1
                    self.last_ids = t.ids
                    self.since = self.clock()
                    self.cv.notify_all()
                    return True
                if left <= 0:
                    return False
                # the lock may be held by work outside this scheduler, which does not notify: look again soon
                self.cv.wait(min(left, 0.05))

    def release(self, used_s: float | None = None):
        """The turn is over: the time it was held (or `used_s`) counts against the conversation's slice."""
        with self.cv:
            if used_s is None:
                used_s = self.clock() - self.since if self.since is not None else 0.0
            self.used += max(0.0, float(used_s))
            self.since = None
            self.lock.release()
            self.cv.notify_all()

    def view(self) -> dict:
        with self.cv:
            return {"slice_s": self.slice_s, "used_s": round(self.used, 1), "waiting": len(self.waiting),
                    "switches": self.switches, "kept": self.kept}


def turns_from_config(cfg: dict, lock) -> tuple[TurnScheduler | TurnPolicy | None, str | None]:
    """The config's "turn_slice_s" (0 or absent: off) and "turn_max_wait_s" (default 120; 0: no limit) -> (the single
    turn's TurnScheduler, or with "parallel" the batch admissions' TurnPolicy, or None; a line to print or None).
    ValueError for a value of the wrong kind."""
    def seconds(key, default, what):
        v = cfg.get(key, default)
        if v is None:
            v = default
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
            raise ValueError(f'"{key}" must be a number of seconds ({what}), not {v!r}')
        return float(v)
    slice_s = seconds("turn_slice_s", 0, "0 = off")
    wait = seconds("turn_max_wait_s", 120, "0 = no limit")
    if not slice_s:
        return None, None
    limit = f"; a request waits at most {wait:g} s" if wait else ""
    if cfg.get("parallel", 1) not in (None, 1):
        # batch slots: the engine orders its admissions with the policy (StrataEngine._admit_next)
        return (TurnPolicy(slice_s, wait),
                f"a batch slot's conversation keeps its slot for {slice_s:g} s of engine time before a new one may "
                f"push it out (turn_slice_s{limit})")
    return (TurnScheduler(lock, slice_s, wait),
            f"the conversation in the engine keeps the turn for {slice_s:g} s of engine time before another waiting "
            f"one takes it (turn_slice_s{limit})")
