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

Priorities (one request at a time; "priority": false turns them off).  A request may say how urgent it is:
"strata_priority" in its body or an X-Strata-Priority header, -20 to 19 like Unix nice, lower first (default
"priority_default", 0; a claim better than "priority_floor" is raised to it).  Its conversation is a session: the
one "strata_session" (or X-Strata-Session, or Anthropic's metadata.user_id) names, else the one whose calls its tool
results answer, else a new one; the session's priority is its latest request's.  The scheduler is a fair one in the
manner of CFS, in bands:
- each session's virtual time grows by the engine time it uses times 1.25^priority (a session 3 better gets about
  twice the share), and a session that comes back after a pause starts no lower than the least of those waiting, so a
  pause banks no credit;
- the band is every request less than "priority_preempt_delta" (default 5) worse than the best one waiting (aged, see
  below) or the session the turn is reserved for: the rest wait.  Within the band, a request that continues the lane
  within its slice goes first (as above), else the least virtual time;
- once a response of a session made tool calls, the turn is reserved for its follow-up: the requests outside its band
  wait until it ends a response with no calls (its turn is over), its calls are ones it does not hold for (the skipped
  tools, a long timeout: it is yielding), or "priority_reserve_s" (default 60) runs out.  A request of the band waits
  for the follow-up only as long as the hold (turn_hold_s) says;
- a request that is a band better than the running one stops it ("priority_preempt"): "full" once the running one has
  had the turn for "priority_min_run_s" (default 2), "timeout" once the better one has waited
  "priority_preempt_timeout_s" (default 10), "off" never (it waits for the request to end).  The stopped request
  gives the turn back and waits again; when it gets it back the engine goes on from what it had read and written (its
  conversation parked meanwhile), and its client sees only a pause;
- a waiting request gains a step of priority for every "priority_aging_s" (default 30; 0: off) it waits, so a
  session in a worse band still gets the turn in the end; it is not stopped by one less than a band better than the
  priority it took the turn with.  turn_max_wait_s applies within the band.

Model variants (the config's "models", serve/models.py).  Each request names the variant it wants (None: whichever is
loaded), and a switch costs far more than a conversation switch: the engine is stopped and the other variant started
(about a minute) - so a request for another variant does not take the turn by arrival or virtual time.  Only the
requests for the loaded variant are eligible, until the scheduler commits to a switch, when:
- nobody waits for the loaded variant, the turn is free and not held for a follow-up, and the loaded variant has had
  the engine for "model_min_resident_s" (default 30); or
- a request for another variant has waited "model_max_wait_s" (default 120); or
- with priorities, a request for another variant is a band better than every request for the loaded one (and the
  running one), once the loaded variant has had its "model_min_resident_s".
Once committed, only the requests for the target variant are eligible: the running request ends, the next one switches
the engine (its session is charged the switch's time), and every request queued for that variant follows.  A running
request is never stopped for another variant unless "model_preempt" is true.
"""
from __future__ import annotations

import collections
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
PRIORITY_MIN, PRIORITY_MAX = -20, 19
PREEMPT_MODES = ("full", "timeout", "off")
SESSION_IDLE_S = 3600.0         # a named session not seen for this long is forgotten


@dataclass(eq=False)
class Session:
    """A conversation as the priorities see it: its priority (its latest request's) and its virtual engine time."""
    key: object = None
    priority: int = 0
    vruntime: float = 0.0
    seen: float = 0.0


@dataclass(eq=False)
class Turn:
    """One response as the turn slice sees it: the IDs of the calls its tool results answer, and the IDs of the calls
    it makes (filled while it runs, so a lane's set is complete by the time its follow-up can arrive)."""
    cites: frozenset = frozenset()
    issued: set = field(default_factory=set)
    tools: set = field(default_factory=set)       # the names of the tools it called
    timeout_s: float = 0.0                        # the longest "timeout" one of its calls asked for (call_timeout_s)
    priority: int | None = None                   # the request's strata_priority (None: its session's, or the default)
    session_key: str | None = None                # the request's strata_session (None: found by its tool results)
    session: Session | None = None                # set by the scheduler as the request enters
    model: str | None = None                      # the model variant it wants (serve/models.py); None: the loaded one


def _header(headers, name):
    try:
        return headers.get(name) if headers is not None else None
    except Exception:
        return None


def request_priority(req: dict, headers=None) -> int | None:
    """The request's "strata_priority" (or its X-Strata-Priority header): an integer, lower first.  None when it gives
    none; ValueError for one that is not a whole number."""
    v = req.get("strata_priority") if isinstance(req, dict) else None
    if v is None:
        v = _header(headers, "X-Strata-Priority")
        if v is None:
            return None
        try:
            v = int(str(v).strip())
        except ValueError:
            raise ValueError(f"X-Strata-Priority must be a whole number (lower first), not {v!r}") from None
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if not isinstance(v, int) or isinstance(v, bool):
        raise ValueError(f"strata_priority must be a whole number (lower first), not {v!r}")
    return v


def request_session(req: dict, headers=None) -> str | None:
    """The session the request names: "strata_session", the X-Strata-Session header, or Anthropic's metadata.user_id
    (Claude Code puts its session's ID in it).  None: the scheduler finds it by the request's tool results."""
    if not isinstance(req, dict):
        return None
    v = req.get("strata_session")
    if v is None:
        v = _header(headers, "X-Strata-Session")
    if v is None and isinstance(req.get("metadata"), dict):
        v = req["metadata"].get("user_id")
    if v is None:
        return None
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"strata_session must be a non-empty string, not {v!r}")
    return v.strip()[:256]


@dataclass
class Priorities:
    """The config's "priority_*" settings (turns_from_config)."""
    delta: int = 5                  # priority_preempt_delta: how much better a band is
    preempt: str = "full"           # priority_preempt: "full", "timeout" or "off"
    preempt_timeout_s: float = 10.0  # priority_preempt_timeout_s: "timeout" stops the running request after this wait
    min_run_s: float = 2.0          # priority_min_run_s: "full" leaves a request the turn at least this long
    reserve_s: float = 60.0         # priority_reserve_s: the longest a session's reservation keeps other bands out
    aging_s: float = 30.0           # priority_aging_s: one step better per this long waiting (0: no aging)
    default: int = 0                # priority_default: a request that gives none, in a session that gave none
    floor: int = PRIORITY_MIN       # priority_floor: the best priority a request may claim

    def clamp(self, p: int) -> int:
        return max(self.floor, min(PRIORITY_MAX, int(p)))

    @staticmethod
    def share(p: int) -> float:
        """Virtual seconds per engine second at priority `p` (CFS's weights: 1.25 per step)."""
        return 1.25 ** p


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


@dataclass(eq=False)
class Ticket:
    turn: Turn
    arrived: float = field(default_factory=time.monotonic)
    seq: int = 0
    preempt: threading.Event = field(default_factory=threading.Event)    # set: a better band waits, give the turn back
    preemptible: bool = True        # False: never stopped for a better band (its state would not be kept)
    run_p: int = 0                  # the priority (aged) it took the turn with


class TurnScheduler:
    """One request at a time: orders the requests that wait for the engine's single turn (`lock`, which other work -
    an image encode, a VRAM resize - may also take directly).  One lane, the engine's session.  acquire() returns True
    once the ticket holds `lock`; release() gives it back and charges the time it was held.  With `prio`
    (Priorities), the bands, the fair order, the reservations and the preemption of the module's docstring; a
    preempted ticket gives the turn back with release(preempted=True) and waits again with requeue()."""

    def __init__(self, lock, slice_s: float = 0.0, max_wait_s: float = 120.0, hold_s: float = 0.0,
                 skip_tools=(), skip_timeout_s: float = 0.0, clock=time.monotonic, prio: Priorities | None = None):
        self.lock = lock
        self.policy = TurnPolicy(slice_s, max_wait_s, clock)
        self.slice_s, self.max_wait_s, self.clock = self.policy.slice_s, self.policy.max_wait_s, clock
        self.hold_s, self.skip_tools = max(0.0, float(hold_s or 0)), frozenset(skip_tools)
        self.skip_timeout_s = max(0.0, float(skip_timeout_s or 0))     # 0: a call's timeout is not looked at
        self.prio = prio
        # the model variants' switching state (serve/models.py's Models: active, since, committed, the policy): set by
        # the server with the config's "models"; None: one model
        self.models = None
        self.sessions: dict = {}            # the named sessions, by key
        self.recent: collections.deque = collections.deque(maxlen=256)  # recent turns: a follow-up finds its session
        self.min_v = 0.0                    # the least virtual time of the sessions waiting (never goes back)
        self.running: Ticket | None = None  # the ticket that holds the turn
        self.no_hold = False                # the last holder was preempted: nothing is held or reserved for it
        self.preemptions = 0                # running requests stopped for a better band
        self.reserve_idle_s = 0.0           # the engine stood idle in a reservation while a request waited
        self.reserve_idle_since: float | None = None
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

    def enter(self, turn: Turn, preemptible: bool = True) -> Ticket:
        t = Ticket(turn, self.clock(), next(self.seq), preemptible=preemptible)
        with self.cv:
            if self.prio is not None:
                self._attach(turn)
            self.waiting.append(t)
            self._check_preempt()
        return t

    def requeue(self, t: Ticket):
        """A ticket that was preempted (and gave the turn back with release(preempted=True)) waits again: its aging and
        its turn_max_wait_s count from now, its place among equals stays its arrival's."""
        with self.cv:
            t.preempt.clear()
            t.arrived = self.clock()
            if t not in self.waiting:
                self.waiting.append(t)
            self._check_preempt()
            self.cv.notify_all()

    def _attach(self, turn: Turn) -> Session:
        """The session of a request entering (the caller holds self.cv): the one it names, else the one whose calls
        its tool results answer, else a new one.  Its priority becomes the request's; one that waits for nothing and
        holds nothing starts no lower than the least virtual time of those waiting."""
        p, now = self.prio, self.clock()
        s = turn.session
        if s is None and turn.session_key is not None:
            s = self.sessions.get(turn.session_key)
        if s is None:
            s = next((x.session for x in reversed(self.recent)
                      if x.session is not None and TurnPolicy.continues(turn, x)), None)
        if s is None:
            s = Session(turn.session_key, p.clamp(p.default))
        if turn.session_key is not None and s.key is None:
            s.key = turn.session_key
        if s.key is not None:
            self.sessions[s.key] = s
        if turn.priority is not None:
            s.priority = p.clamp(turn.priority)
        busy = (self.running is not None and self.running.turn.session is s) or \
            any(w.turn.session is s for w in self.waiting)
        if not busy:
            s.vruntime = max(s.vruntime, self.min_v)
        s.seen = now
        turn.session = s
        self.recent.append(turn)
        if len(self.sessions) > 64:                     # forget the named sessions long gone
            for k in [k for k, x in self.sessions.items() if now - x.seen > SESSION_IDLE_S]:
                del self.sessions[k]
        return s

    def _eligible(self, t: Ticket, model: str | None) -> bool:
        return t.turn.model is None or t.turn.model == model

    def _by_model(self, order: list[Ticket]) -> list[Ticket]:
        """The waiting tickets that may take the turn with model variants (the module's docstring; the caller holds
        self.cv): the loaded variant's, or once a switch is committed the target's.  Commits a switch when its time
        has come."""
        m = self.models
        if m is None or not order:
            return order
        if m.committed is not None:
            target = [t for t in order if self._eligible(t, m.committed)]
            if target:
                return target
            m.committed = None                          # everyone who wanted it left
        same = [t for t in order if self._eligible(t, m.active)]
        others = [t for t in order if not self._eligible(t, m.active)]
        if not others:
            return same
        now = self.clock()
        resident = now - m.since >= m.min_resident_s or not m.loaded()    # an unloaded engine costs nothing to keep
        first = min(others, key=lambda t: t.arrived)
        pick = None
        if m.max_wait_s > 0 and now - first.arrived >= m.max_wait_s:
            pick = first
        elif resident and not same and self.since is None and not self.holds():
            pick = first
        elif resident and self.prio is not None:
            best = min(others, key=lambda t: (self.effective(t, now), t.seq))
            mine = [self.effective(t, now) for t in same] + ([self.running.run_p] if self.running is not None else [])
            if mine and self.effective(best, now) <= min(mine) - self.prio.delta:
                pick = best
        if pick is None:
            return same
        m.committed = pick.turn.model
        return [t for t in order if self._eligible(t, m.committed)]

    def model_switched(self):
        """The engine now runs another variant (the server switched it while holding the turn): no conversation of the
        old one continues in it."""
        with self.cv:
            self.lane, self.used = None, 0.0
            self.cv.notify_all()

    def effective(self, t: Ticket, now: float | None = None) -> int:
        """A waiting ticket's priority: its session's, a step better for every priority_aging_s it has waited."""
        p = t.turn.session.priority if t.turn.session is not None else self.prio.default
        if self.prio.aging_s > 0:
            now = self.clock() if now is None else now
            p -= int(max(0.0, now - t.arrived) // self.prio.aging_s)
        return max(PRIORITY_MIN, p)

    def reserved(self, now: float | None = None) -> Session | None:
        """The session the free turn is reserved for (the caller holds self.cv): the last response made tool calls it
        is not yielding on (none to a skipped tool, none with a long timeout), it was not preempted, and
        priority_reserve_s has not run out since it gave the turn back."""
        p, lane = self.prio, self.lane
        if p is None or p.reserve_s <= 0 or self.released is None or self.no_hold or lane is None or \
                lane.session is None or not lane.issued or lane.tools & self.skip_tools or \
                (self.skip_timeout_s and lane.timeout_s > self.skip_timeout_s):
            return None
        now = self.clock() if now is None else now
        return lane.session if now - self.released < p.reserve_s else None

    def _check_preempt(self):
        """Asks the running ticket to give the turn back when a request a band better waits (the caller holds
        self.cv): at once ("full", after priority_min_run_s), once that one has waited priority_preempt_timeout_s
        ("timeout"), or never ("off")."""
        p, r = self.prio, self.running
        if p is None or p.preempt == "off" or r is None or not r.preemptible or r.preempt.is_set() or not self.waiting:
            return
        now = self.clock()
        if p.preempt == "full" and self.since is not None and now - self.since < p.min_run_s:
            return
        m = self.models
        for t in self.waiting:
            if m is not None and not m.preempt and not self._eligible(t, m.active):
                continue                                # a variant switch waits for the running request to end
            if self.effective(t, now) <= r.run_p - p.delta and \
                    (p.preempt == "full" or now - t.arrived >= p.preempt_timeout_s):
                r.preempt.set()
                self.preemptions += 1
                return

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
        return self.released is not None and not self.no_hold and self.hold_wanted() and \
            self.clock() - self.released < self.hold_s

    def next_ticket(self) -> Ticket | None:
        """The ticket whose turn it is (the caller holds self.cv)."""
        order = self._by_model(sorted(self.waiting, key=lambda t: t.seq))
        if self.prio is not None:
            return self._next_by_priority(order)
        if order and self.holds():
            # only the follow-up takes a held turn - or a request that has waited turn_max_wait_s
            if self.max_wait_s > 0 and self.clock() - order[0].arrived >= self.max_wait_s:
                return order[0]
            return next((t for t in order if self.continues(t.turn)), None)
        lanes = [(self.lane, self.used)] if self.lane is not None else []
        return self.policy.pick([(t, t.turn, t.arrived) for t in order], lanes, room=not lanes)

    def _next_by_priority(self, order) -> Ticket | None:
        """next_ticket with priorities: the band (those less than priority_preempt_delta worse than the best waiting,
        or than the session the turn is reserved for), then within it the hold, turn_max_wait_s, the lane within its
        slice, and the least virtual time."""
        if not order:
            return None
        now, delta = self.clock(), self.prio.delta
        eff = [(t, self.effective(t, now)) for t in order]
        res = self.reserved(now)
        best = min([e for _, e in eff] + ([res.priority] if res is not None else []))
        band = [t for t, e in eff if e < best + delta]
        if not band:
            return None                                 # the turn is reserved for a better band
        lane_p = self.lane.session.priority if self.lane is not None and self.lane.session is not None else None
        if self.holds() and (lane_p is None or lane_p < best + delta):
            # the lane's own band waits for its follow-up for the hold (a better band does not)
            if self.max_wait_s > 0 and now - band[0].arrived >= self.max_wait_s:
                return band[0]
            return next((t for t in band if self.continues(t.turn)), None)
        if self.max_wait_s > 0 and now - band[0].arrived >= self.max_wait_s:
            return band[0]
        if self.slice_s > 0 and self.lane is not None and self.used < self.slice_s:
            follow = next((t for t in band if self.continues(t.turn)), None)
            if follow is not None:
                return follow
        return min(band, key=lambda t: (t.turn.session.vruntime if t.turn.session is not None else 0.0, t.seq))

    def acquire(self, t: Ticket, timeout: float) -> bool:
        """Waits up to `timeout` seconds for `t`'s turn and the lock; False on timeout (the caller beats and retries)."""
        end = self.clock() + timeout
        with self.cv:
            while True:
                left = end - self.clock()
                held = self.holds()
                nxt = self.next_ticket()
                if self.prio is not None:
                    self._check_preempt()                   # a wait (priority_preempt_timeout_s, aging) has grown
                    if nxt is None and self.waiting and self.reserve_idle_since is None and \
                            self.reserved() is not None and not self.lock.locked():
                        self.reserve_idle_since = self.clock()  # everyone waiting is kept out by a reservation
                if nxt is t and self.lock.acquire(blocking=False):
                    self.waiting.remove(t)
                    self._count_hold(t, held)
                    if self.prio is not None:
                        now = self.clock()
                        if self.reserve_idle_since is not None:
                            self.reserve_idle_s += now - self.reserve_idle_since
                            self.reserve_idle_since = None
                        t.run_p = self.effective(t, now)
                        t.preempt.clear()
                        self.running, self.no_hold = t, False
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

    def release(self, used_s: float | None = None, preempted: bool = False):
        """The turn is over: the time it was held (or `used_s`) counts against the conversation's slice, and with
        priorities against its session's virtual time.  `preempted`: it gave the turn back for a better band (it will
        requeue()): no hold and no reservation for it."""
        with self.cv:
            if used_s is None:
                used_s = self.clock() - self.since if self.since is not None else 0.0
            used_s = max(0.0, float(used_s))
            self.used += used_s
            r = self.running
            if self.prio is not None and r is not None and r.turn.session is not None:
                s = r.turn.session
                s.vruntime += used_s * self.prio.share(s.priority)
                waiting = [w.turn.session.vruntime for w in self.waiting if w.turn.session is not None]
                self.min_v = max(self.min_v, min(waiting + [s.vruntime]))
            self.running, self.no_hold = None, bool(preempted)
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
                    "hold_idle_s": round(self.hold_idle_s, 1),
                    **({"priority": self._priority_view()} if self.prio is not None else {}),
                    **({"models": self._models_view()} if self.models is not None else {})}

    def _models_view(self) -> dict:
        """view()'s model variants (the caller holds self.cv): what waits for each one, and the committed switch."""
        m, waiting = self.models, {}
        for t in self.waiting:
            k = t.turn.model or m.active
            waiting[k] = waiting.get(k, 0) + 1
        return {"active": m.active, "committed": m.committed, "waiting": waiting}

    def _priority_view(self) -> dict:
        """view()'s priorities (the caller holds self.cv)."""
        p, now = self.prio, self.clock()
        res = self.reserved(now)
        bands: dict = {}
        for t in self.waiting:
            e = self.effective(t, now)
            bands[e] = bands.get(e, 0) + 1
        return {"preempt": p.preempt, "delta": p.delta, "min_run_s": p.min_run_s,
                "preempt_timeout_s": p.preempt_timeout_s, "reserve_s": p.reserve_s, "aging_s": p.aging_s,
                "running": self.running.run_p if self.running is not None else None,
                "reserved_for": res.priority if res is not None else None,
                "waiting": {str(k): v for k, v in sorted(bands.items())},
                "preemptions": self.preemptions, "reserve_idle_s": round(self.reserve_idle_s, 1),
                "sessions": len({id(t.session) for t in self.recent if t.session is not None})}


def turns_from_config(cfg: dict, lock, models: bool = False) -> tuple[TurnScheduler | TurnPolicy | None, str | None]:
    """The config's "turn_slice_s" (0 or absent: off), "turn_max_wait_s" (default 120; 0: no limit), "turn_hold_s"
    (default 3; 0: off; one request at a time only), "turn_hold_skip_tools" (default HOLD_SKIP_TOOLS) and
    "turn_hold_skip_timeout_s" (default 120; 0: a call's timeout is not looked at), and the priorities (the module's
    docstring): "priority" (default true with a turn slice; true without one turns the scheduler on for them alone),
    "priority_preempt" ("full", "timeout" or "off"), "priority_preempt_delta", "priority_min_run_s",
    "priority_preempt_timeout_s", "priority_reserve_s" (0: no reservations), "priority_aging_s" (0: no aging),
    "priority_default" and "priority_floor" -> (the single turn's TurnScheduler, or with "parallel" the batch
    admissions' TurnPolicy (no priorities), or None; a line to print or None).  `models`: the config has model
    variants, which the scheduler switches between, so it is on without a slice or priorities too.  ValueError for a
    value of the wrong kind."""
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
    prio_on = cfg.get("priority")
    if prio_on is not None and not isinstance(prio_on, bool):
        raise ValueError(f'"priority" must be true or false, not {prio_on!r}')
    prio = priorities_from_config(cfg) if prio_on is True or (prio_on is None and slice_s) else None
    if not slice_s and prio_on is not True and not models:
        return None, None
    limit = f"; a request waits at most {wait:g} s" if wait else ""
    if cfg.get("parallel", 1) not in (None, 1):
        # batch slots: the engine orders its admissions with the policy (StrataEngine._admit_next)
        return (TurnPolicy(slice_s, wait),
                f"a batch slot's conversation keeps its slot for {slice_s:g} s of engine time before a new one may "
                f"push it out (turn_slice_s{limit})")
    held = f"; after tool calls the turn waits up to {hold:g} s for the follow-up (turn_hold_s)" if hold else ""
    if slice_s:
        note = (f"the conversation in the engine keeps the turn for {slice_s:g} s of engine time before another "
                f"waiting one takes it (turn_slice_s{limit}){held}")
    elif prio_on is True:
        note = f"the waiting requests take the turn by priority (no turn slice{limit})"
    else:
        note = f"the waiting requests take the turn in arrival order, switching model variants by policy{limit}"
    if prio is not None:
        stop = {"full": f"stops a running one {prio.delta} worse (after it has had {prio.min_run_s:g} s)",
                "timeout": f"stops a running one {prio.delta} worse once it has waited {prio.preempt_timeout_s:g} s",
                "off": f"goes before those {prio.delta} worse once the running request ends"}[prio.preempt]
        aging = f", a step better per {prio.aging_s:g} s waited" if prio.aging_s else ", no aging"
        reserve = f"; a session's tool calls keep worse bands out for up to {prio.reserve_s:g} s" \
            if prio.reserve_s else ""
        note += (f"; priorities: a request {stop} (priority_preempt: {prio.preempt}){reserve}{aging}")
    return TurnScheduler(lock, slice_s, wait, hold, skip, skip_timeout, prio=prio), note


def priorities_from_config(cfg: dict) -> Priorities:
    """The config's "priority_*" keys -> Priorities.  ValueError for a value of the wrong kind."""
    d = Priorities()

    def number(key, default, what, whole=False, low=0):
        v = cfg.get(key, default)
        if v is None:
            v = default
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v < low or (whole and v != int(v)):
            raise ValueError(f'"{key}" must be {what}, not {v!r}')
        return int(v) if whole else float(v)
    mode = cfg.get("priority_preempt", d.preempt)
    if mode is True:
        mode = "full"
    elif mode is False:
        mode = "off"
    if mode not in PREEMPT_MODES:
        raise ValueError(f'"priority_preempt" must be one of {", ".join(PREEMPT_MODES)}, not {mode!r}')
    floor = number("priority_floor", d.floor, f"a whole number from {PRIORITY_MIN} to {PRIORITY_MAX}", True,
                   PRIORITY_MIN)
    default = number("priority_default", d.default, f"a whole number from {PRIORITY_MIN} to {PRIORITY_MAX}", True,
                     PRIORITY_MIN)
    if floor > PRIORITY_MAX or default > PRIORITY_MAX:
        raise ValueError(f'"priority_floor" and "priority_default" go from {PRIORITY_MIN} to {PRIORITY_MAX}')
    return Priorities(
        delta=number("priority_preempt_delta", d.delta, "a whole number of steps, 1 or more", True, 1),
        preempt=mode,
        preempt_timeout_s=number("priority_preempt_timeout_s", d.preempt_timeout_s, "a number of seconds"),
        min_run_s=number("priority_min_run_s", d.min_run_s, "a number of seconds"),
        reserve_s=number("priority_reserve_s", d.reserve_s, "a number of seconds (0 = no reservations)"),
        aging_s=number("priority_aging_s", d.aging_s, "a number of seconds (0 = no aging)"),
        default=max(floor, default), floor=floor)
