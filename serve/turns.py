"""serve/turns.py - which waiting request the engine takes next, with a time slice per conversation (the config's
"turn_slice_s", opt-in).

Every switch to another conversation has a cost: one request at a time, the engine parks the outgoing conversation and
reads the incoming one back (with --conversation-cache-dir: a file written and a file read, hundreds of MB each); with
"parallel" (batch slots), a new conversation in a slot pushes out the one that slot held, whose next turn then reads
its history again.  When several clients take turns - an agent and its subagents, two harnesses - their requests
interleave and nearly every request is such a switch.

The conversations the engine holds are its lanes: the single session, or each batch slot's.  A lane keeps its place
while it has used less than the slice of engine time since it took the lane (each request's time on the engine, from
taking the turn or slot to giving it back): a waiting request that continues a lane within its slice goes before one
that would push a conversation out.

A request continues a lane when it is the next step of the response the lane last ran: its tool results answer calls
that response made.  Every call's ID is minted here (serve/frontend.py's ToolCall, 96 random bits) and a client sends
it back with the result (Chat Completions' tool_call_id, Anthropic's tool_use_id, the Responses API's call_id), so the
match is exact: no prompt comparison, no shared system prompt mistaken for the same conversation, no minimum length.
A new user message starts a new turn of its own (a switch); so does a retry, which cites nothing new.

A request that pushes nobody out (it continues a lane, or an empty lane is free) goes in arrival order; so does
everything once the lanes it would wait for are past their slice; and one that has waited "turn_max_wait_s" goes next
whatever the slices say, so nobody starves.  A turn changes hands between requests only: a request is never stopped for
the slice.  Without a slice: arrival order.

One request at a time, a response's follow-up can only arrive after the response has ended - and a request already
waiting would take the turn at once.  So when a response that made tool calls gives the turn back within its slice,
the turn is held for its follow-up for "turn_hold_s" (default 3): only a request that answers those calls takes it
meanwhile.  Most tools come back well inside that (a file read, an edit, a quick command: under a second), and a
switch costs seconds (the park, then the tail since the checkpoint read again on the way back); a long command or a
web fetch outlasts the hold and the turn goes on as before.  Calls to "turn_hold_skip_tools" (HOLD_SKIP_TOOLS) are
not waited for: a subagent's own requests are what its call waits on (Claude Code's Agent/Task, Hermes Agent's
delegate_task), a question waits for a person (AskUserQuestion, clarify), and Claude Code's web calls almost never
return in time.  Nor is a call whose timeout argument is above "turn_hold_skip_timeout_s" (default 120): the model asks
for a long timeout when it expects a long command (in Claude Code transcripts, 9 in 10 Bash calls without one returned
within 3 s, 1 in 5 with one over 2 minutes; pi's bash: 91% without one, 53% with one).  The unit is read from the
tool's own schema in the request (Claude Code's Bash: ms; pi's bash and Hermes Agent's terminal: seconds); a call that
runs in the background returns at once whatever its timeout.

Time, not tokens: what the other clients wait is time, and a switch costs about a fixed time (a park and a read-back),
so a slice of T seconds bounds the switching overhead to that cost over T.  A token count would weigh a prompt read
(~1-2K tokens/s) and a generated token (~30/s) alike.
"""
from __future__ import annotations

import itertools
import re
import threading
import time
from dataclasses import dataclass, field

# the tool results' item and block types that carry the ID of the call they answer, per API
OUTPUTS = ("function_call_output", "custom_tool_call_output")
MODEL_ITEMS = ("function_call", "custom_tool_call", "reasoning")
# tools whose results the turn is not held for: a subagent's call waits on the subagent's own requests (holding the
# turn for it would make them wait), a question waits for a person, and Claude Code's web calls almost never return
# within the hold (pi's web_search and fetch_content mostly do: 96% and 74% within 3 s)
HOLD_SKIP_TOOLS = (
    "Agent", "Task", "AskUserQuestion", "ExitPlanMode", "WebFetch", "WebSearch",    # Claude Code
    "delegate_task", "clarify", "kanban_block",                                     # Hermes Agent
)
# a call's run time limit, by argument name: (name, seconds per unit, or None to read the unit from the schema)
TIMEOUT_KEYS = (("timeout", None), ("timeout_ms", 0.001), ("timeout_s", 1.0), ("timeout_sec", 1.0),
                ("timeout_seconds", 1.0))
BACKGROUND_KEYS = ("run_in_background", "background")   # Claude Code's Bash; Hermes Agent's terminal


@dataclass(eq=False)
class Turn:
    """One response as the turn slice sees it: the IDs of the calls its tool results answer, and the IDs of the calls
    it makes (filled while it runs, so a lane's set is complete by the time its follow-up can arrive)."""
    cites: frozenset = frozenset()
    issued: set = field(default_factory=set)
    tools: set = field(default_factory=set)       # the names of the tools it called
    timeout_s: float = 0.0                        # the longest "timeout" one of its calls asked for (call_timeout_s)


def _unit_from_text(text) -> float | None:
    """Seconds per unit from a schema description: "...in milliseconds" -> 0.001, "...in seconds" -> 1."""
    text = (text or "").lower() if isinstance(text, str) else ""
    if "millisecond" in text or re.search(r"\bms\b", text):
        return 0.001
    if "second" in text:
        return 1.0
    return None


def timeout_units(tools) -> dict:
    """{tool name: (argument, seconds per unit)} for each tool in the request's list (the template's form: name,
    description, parameters) with a timeout argument.  The unit comes from the argument's name, else its description,
    else the tool's: Claude Code's Bash says milliseconds, pi's bash and Hermes Agent's terminal say seconds."""
    units = {}
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        props = (t.get("parameters") or {}).get("properties") if isinstance(t.get("parameters"), dict) else None
        if not isinstance(props, dict):
            continue
        for key, unit in TIMEOUT_KEYS:
            if key in props:
                prop = props[key] if isinstance(props[key], dict) else {}
                unit = unit or _unit_from_text(prop.get("description")) or _unit_from_text(t.get("description"))
                units[t.get("name")] = (key, unit)
                break
    return units


def call_timeout_s(arguments, unit=None) -> float:
    """The run time a call asks for, in seconds; 0 when it gives none or runs in the background (it returns at once).
    `unit`: (argument, seconds per unit) from timeout_units; without it, "timeout_ms" is milliseconds and a "timeout"
    of 1000 or more is taken for milliseconds, below that for seconds."""
    if not isinstance(arguments, dict) or any(arguments.get(k) is True for k in BACKGROUND_KEYS):
        return 0.0
    keys = [unit] if unit else [(k, u) for k, u in TIMEOUT_KEYS]
    for key, per in keys:
        v = arguments.get(key)
        if isinstance(v, str):
            try:
                v = float(v)
            except ValueError:
                continue
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            return float(v) * (per or (0.001 if v >= 1000 else 1.0))
    return 0.0


def cited_calls(path: str, req: dict) -> frozenset:
    """The call IDs that the request's tool results answer: those after the model's last turn in its history (a
    client may add a user message after the results, Claude Code a reminder block beside them).  `path` names the API:
    /v1/chat/completions, /v1/messages or /v1/responses.  Empty when its history ends with no tool result."""
    ids = set()
    if path == "/v1/responses":
        items = req.get("input")
        for item in reversed(items if isinstance(items, list) else []):
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind in MODEL_ITEMS or (kind in (None, "message") and item.get("role") == "assistant"):
                break
            if kind in OUTPUTS and isinstance(item.get("call_id"), str):
                ids.add(item["call_id"])
        return frozenset(ids)
    messages = req.get("messages")
    for m in reversed(messages if isinstance(messages, list) else []):
        if not isinstance(m, dict):
            continue
        if m.get("role") == "assistant":
            break
        if path == "/v1/messages":
            for block in m.get("content") if isinstance(m.get("content"), list) else []:
                if isinstance(block, dict) and block.get("type") == "tool_result" and \
                        isinstance(block.get("tool_use_id"), str):
                    ids.add(block["tool_use_id"])
        elif m.get("role") == "tool" and isinstance(m.get("tool_call_id"), str):
            ids.add(m["tool_call_id"])
    return frozenset(ids)


class TurnPolicy:
    """The order alone, over any number of lanes: pick() is asked with the waiting requests and the lanes as they are."""

    def __init__(self, slice_s: float = 0.0, max_wait_s: float = 120.0, clock=time.monotonic):
        self.slice_s = max(0.0, float(slice_s or 0))
        self.max_wait_s = max(0.0, float(max_wait_s or 0))
        self.clock = clock

    @staticmethod
    def continues(turn: Turn | None, lane: Turn | None) -> bool:
        """`turn` is the next step of the response `lane` ran: it answers a call that response made (or it is that
        same response, in a later pass)."""
        if turn is None or lane is None:
            return False
        return turn is lane or bool(turn.cites & lane.issued)

    def pick(self, waiting, lanes, room: bool):
        """The waiting request that goes next.  `waiting`: (key, Turn, arrived) in arrival order; `lanes`: (the Turn
        it last ran, seconds used) of each conversation the engine holds; `room`: an empty lane is free (a new conversation pushes
        nobody out).  None when nobody waits."""
        if not waiting:
            return None
        key, turn, arrived = waiting[0]
        if self.slice_s <= 0 or room or (self.max_wait_s > 0 and self.clock() - arrived >= self.max_wait_s):
            return key
        if any(self.continues(turn, lane) for lane, _ in lanes):
            return key                                  # the oldest pushes nobody out
        for k, w_turn, _ in waiting:                    # the oldest that continues a lane within its slice
            if any(used < self.slice_s and self.continues(w_turn, lane) for lane, used in lanes):
                return k
        return key


@dataclass
class Ticket:
    turn: Turn
    arrived: float = field(default_factory=time.monotonic)
    seq: int = 0


class TurnScheduler:
    """One request at a time: orders the requests that wait for the engine's single turn (`lock`, which other work -
    an image encode, a VRAM resize - may also take directly).  One lane, the engine's session.  acquire() returns True
    once the ticket holds `lock`; release() gives it back and charges the time it was held."""

    def __init__(self, lock, slice_s: float = 0.0, max_wait_s: float = 120.0, hold_s: float = 0.0,
                 skip_tools=(), skip_timeout_s: float = 0.0, clock=time.monotonic):
        self.lock = lock
        self.policy = TurnPolicy(slice_s, max_wait_s, clock)
        self.slice_s, self.max_wait_s, self.clock = self.policy.slice_s, self.policy.max_wait_s, clock
        self.hold_s, self.skip_tools = max(0.0, float(hold_s or 0)), frozenset(skip_tools)
        self.skip_timeout_s = max(0.0, float(skip_timeout_s or 0))     # 0: a call's timeout is not looked at
        self.released: float | None = None  # when the last holder gave the turn back (the hold counts from it)
        self.cv = threading.Condition()
        self.waiting: list[Ticket] = []
        self.seq = itertools.count()
        self.lane: Turn | None = None       # the response that ran last
        self.used = 0.0                     # seconds the current conversation has held the turn since it took it
        self.since: float | None = None     # when the current holder took the turn
        self.switches = 0                   # turns that went to another conversation
        self.kept = 0                       # turns that went to a later arrival of the current conversation
        self.hold_hits = 0                  # a hold kept the turn for its follow-up while another request waited
        self.hold_misses = 0                # a hold ran out and a request that waited behind it took the turn
        self.hold_idle_s = 0.0              # the engine stood idle in a hold while a request waited

    def continues(self, turn: Turn) -> bool:
        return TurnPolicy.continues(turn, self.lane)

    def enter(self, turn: Turn) -> Ticket:
        t = Ticket(turn, self.clock(), next(self.seq))
        with self.cv:
            self.waiting.append(t)
        return t

    def leave(self, t: Ticket):
        """A ticket that gave up waiting (its client left)."""
        with self.cv:
            if t in self.waiting:
                self.waiting.remove(t)
                self.cv.notify_all()

    def hold_wanted(self) -> bool:
        """The response that ran last is worth holding the turn for (the caller holds self.cv): it made tool calls,
        none to a skipped tool or with a long timeout, and its conversation is within its slice.  Read when asked, not at release: a call
        the parser completes as the response ends is counted even when it comes after the release."""
        lane = self.lane
        return (self.hold_s > 0 and lane is not None and bool(lane.issued) and not lane.tools & self.skip_tools
                and not (self.skip_timeout_s and lane.timeout_s > self.skip_timeout_s) and self.used < self.slice_s)

    def holds(self) -> bool:
        """The turn is free but held for the follow-up of the response that gave it back (the caller holds self.cv)."""
        return self.released is not None and self.hold_wanted() and self.clock() - self.released < self.hold_s

    def next_ticket(self) -> Ticket | None:
        """The ticket whose turn it is (the caller holds self.cv)."""
        order = sorted(self.waiting, key=lambda t: t.seq)
        if order and self.holds():
            # only the follow-up takes a held turn - or a request that has waited turn_max_wait_s
            if self.max_wait_s > 0 and self.clock() - order[0].arrived >= self.max_wait_s:
                return order[0]
            return next((t for t in order if self.continues(t.turn)), None)
        lanes = [(self.lane, self.used)] if self.lane is not None else []
        return self.policy.pick([(t, t.turn, t.arrived) for t in order], lanes, room=not lanes)

    def acquire(self, t: Ticket, timeout: float) -> bool:
        """Waits up to `timeout` seconds for `t`'s turn and the lock; False on timeout (the caller beats and retries)."""
        end = self.clock() + timeout
        with self.cv:
            while True:
                left = end - self.clock()
                held = self.holds()
                if self.next_ticket() is t and self.lock.acquire(blocking=False):
                    self.waiting.remove(t)
                    self._count_hold(t, held)
                    if self.continues(t.turn):
                        if self.waiting and t.seq > min(x.seq for x in self.waiting):
                            self.kept += 1
                    else:
                        self.used = 0.0
                        if self.lane is not None:
                            self.switches += 1
                    self.lane = t.turn
                    self.since, self.released = self.clock(), None
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
            self.since, self.released = None, self.clock()
            self.lock.release()
            self.cv.notify_all()

    def _count_hold(self, t: Ticket, held: bool):
        """/metrics, as `t` takes the turn (`held`: the turn was held until now): a hit is the follow-up taking a held
        turn while another request waits; a miss is a request that waited behind a hold taking the turn once it ran
        out.  The idle time is from the release (or the first wait, if later) to now, in a hold that was wanted."""
        if self.released is None or not self.hold_wanted():
            return
        now = self.clock()
        first = min([t.arrived] + [x.arrived for x in self.waiting])    # the longest wait, t's included
        if first < now:
            self.hold_idle_s += max(0.0, min(now, self.released + self.hold_s) - max(self.released, first))
        if self.continues(t.turn):
            if held and self.waiting:
                self.hold_hits += 1
        elif not held and t.arrived < self.released + self.hold_s:
            self.hold_misses += 1

    def view(self) -> dict:
        with self.cv:
            return {"slice_s": self.slice_s, "used_s": round(self.used, 1), "waiting": len(self.waiting),
                    "switches": self.switches, "kept": self.kept, "hold_s": self.hold_s, "holding": self.holds(),
                    "hold_hits": self.hold_hits, "hold_misses": self.hold_misses,
                    "hold_idle_s": round(self.hold_idle_s, 1)}


def turns_from_config(cfg: dict, lock) -> tuple[TurnScheduler | TurnPolicy | None, str | None]:
    """The config's "turn_slice_s" (0 or absent: off), "turn_max_wait_s" (default 120; 0: no limit), "turn_hold_s"
    (default 3; 0: off; one request at a time only), "turn_hold_skip_tools" (default HOLD_SKIP_TOOLS) and
    "turn_hold_skip_timeout_s" (default 120; 0: a call's timeout is not looked at) -> (the
    single turn's TurnScheduler, or with "parallel" the batch admissions' TurnPolicy, or None; a line to print or
    None).  ValueError for a value of the wrong kind."""
    def seconds(key, default, what):
        v = cfg.get(key, default)
        if v is None:
            v = default
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
            raise ValueError(f'"{key}" must be a number of seconds ({what}), not {v!r}')
        return float(v)
    slice_s = seconds("turn_slice_s", 0, "0 = off")
    wait = seconds("turn_max_wait_s", 120, "0 = no limit")
    hold = seconds("turn_hold_s", 3, "0 = off")
    skip_timeout = seconds("turn_hold_skip_timeout_s", 120, "0 = not looked at")
    skip = cfg.get("turn_hold_skip_tools", HOLD_SKIP_TOOLS)
    if skip is None:
        skip = HOLD_SKIP_TOOLS
    if not isinstance(skip, (list, tuple)) or not all(isinstance(x, str) for x in skip):
        raise ValueError(f'"turn_hold_skip_tools" must be a list of tool names, not {skip!r}')
    if not slice_s:
        return None, None
    limit = f"; a request waits at most {wait:g} s" if wait else ""
    if cfg.get("parallel", 1) not in (None, 1):
        # batch slots: the engine orders its admissions with the policy (StrataEngine._admit_next)
        return (TurnPolicy(slice_s, wait),
                f"a batch slot's conversation keeps its slot for {slice_s:g} s of engine time before a new one may "
                f"push it out (turn_slice_s{limit})")
    held = f"; after tool calls the turn waits up to {hold:g} s for the follow-up (turn_hold_s)" if hold else ""
    return (TurnScheduler(lock, slice_s, wait, hold, skip, skip_timeout),
            f"the conversation in the engine keeps the turn for {slice_s:g} s of engine time before another waiting "
            f"one takes it (turn_slice_s{limit}){held}")
