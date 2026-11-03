#!/usr/bin/env python3
"""Records the turn scheduler's demo videos: pi agents in tmux panes beside monitor.py, driven by Strata's /metrics,
recorded with asciinema, rendered with agg and encoded to MP4 (no audio) with ffmpeg.

    run.py [scenario ...] --out scheduler-demo/videos     # scenarios: preempt, yield, same-priority, reservation

The driver marks the moments that show the scheduler at work; the video plays those in real time and fast-forwards
the rest (a "▶▶ n×" badge in the top right says how fast), to keep each video under MAX_S seconds.

Each agent runs `pi --no-session --thinking off` in an empty working directory, with its own pi config directory that
loads only pi-strata (no memory or web extensions), its priority from PI_STRATA_PRIORITY.  Needs tmux, pi,
asciinema, agg and ffmpeg (ASCIINEMA, AGG, FFMPEG override the commands).
"""
import argparse
import json
import os
import re
import shlex
import textwrap
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PI_STRATA = Path.home() / "Projects" / "pi-strata"
USER_AGENT_DIR = Path.home() / ".pi" / "agent"
COLS, ROWS, MONITOR_ROWS, CAPTION_ROWS = 200, 52, 12, 4
MAX_S, HIGHLIGHT_S, LAST_FRAME_S = 28.0, 15.0, 2.0   # the video's length, its real-time part at most, the last frame
DEMO, REC, SOCKET = "strata-demo", "strata-rec", "strata-demo"


class Scenario:
    def __init__(self, name, title, caption, agents):
        self.name, self.title, self.caption, self.agents = name, title, caption, agents   # agents: [(name, prio)]


SCENARIOS = {
    "preempt": Scenario(
        "preempt", "A higher-priority agent preempts a lower-priority one",
        "Agent L (priority 10) is writing a long story. Agent H (priority 0, a band better) asks a quick question: the "
        "scheduler stops L mid-reply, H answers at once, then L goes on from the token it stopped at.",
        [("L", 10), ("H", 0)]),
    "yield": Scenario(
        "yield", "An agent in a long tool call yields the engine",
        "Agent A (priority 0) runs a 30 s command with a 300 s timeout: a long call, so A yields instead of reserving "
        "the engine. Agent B (priority 10) gets the engine meanwhile; when A's command ends, A takes it back.",
        [("A", 0), ("B", 10)]),
    "same-priority": Scenario(
        "same-priority", "Two agents of the same priority: one waits for the other's turn",
        "Agents A and B both have priority 0. A starts a task of several quick tool calls; B asks right after. A keeps "
        "the engine through its tool calls (its time slice, held for each follow-up), and B's request runs once A's "
        "turn is done: no conversation switches in the middle.",
        [("A", 0), ("B", 0)]),
    "reservation": Scenario(
        "reservation", "A short tool call keeps a lower-priority agent waiting",
        "Agent A (priority 0) runs a 10 s command with no timeout: not a long call, so the engine stays reserved for "
        "A's follow-up and Agent B (priority 10) waits, instead of making A read its conversation back after the call.",
        [("A", 0), ("B", 10)]),
}

PROMPTS = {
    ("preempt", "L"): "Write a 900-word short story about a lighthouse keeper who finds a message in a bottle. "
                      "Do not use any tools; just write the story.",
    ("preempt", "H"): "Quick question, answer in one sentence and use no tools: what is the capital of Australia?",
    ("yield", "A"): "Use the bash tool once to run exactly `sleep 30 && echo build finished`, passing a timeout of 300 "
                    "seconds. Then tell me its output in one line.",
    ("yield", "B"): "Write a 1500-word short story about a cartographer mapping an island that keeps changing shape. "
                    "Do not use any tools; just write the story.",
    ("same-priority", "A"): "Use the bash tool for each of these, one call at a time and in this order: `uname -sr`, "
                            "`nproc`, `df -h / | tail -1`, `date +%A`. Then summarise the results in four short lines.",
    ("same-priority", "B"): "Write a 120-word poem about rain. Do not use any tools.",
    ("reservation", "A"): "Use the bash tool to run exactly `sleep 10 && echo step one done` (do not pass a timeout), "
                          "then run `echo step two done` the same way. Then say both steps are done, in one line.",
    ("reservation", "B"): "Write a 300-word poem about autumn leaves. Do not use any tools.",
}


def sh(*args, check=True, **kw):
    return subprocess.run(args, check=check, text=True, capture_output=True, **kw)


def tmux(*args, check=True):
    """tmux on the demo's own server socket: the user's sessions and options are not touched."""
    return sh("tmux", "-L", SOCKET, *args, check=check).stdout


class Metrics:
    def __init__(self, agent_dir):
        base = json.loads((agent_dir / "strata.json").read_text())["baseUrl"]
        self.base = re.sub(r"/v1/?$", "", base.rstrip("/"))
        self.key = json.loads((agent_dir / "auth.json").read_text())["strata"]["key"]

    def get(self):
        req = urllib.request.Request(self.base + "/metrics", headers={"Authorization": f"Bearer {self.key}"})
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                return json.loads(r.read())
        except Exception:
            return {}

    def state(self):
        m = self.get()
        live, turns = m.get("live", {}), m.get("turns", {})
        pr = turns.get("priority") or {}
        return {"state": live.get("state"), "generated": live.get("generated") or 0, "running": pr.get("running"),
                "waiting": sum((pr.get("waiting") or {}).values()), "reserved": pr.get("reserved_for"),
                "preemptions": pr.get("preemptions", 0), "switches": turns.get("switches", 0),
                "done": m.get("totals", {}).get("requests", 0)}

    def wait(self, what, pred, timeout=300):
        end = time.time() + timeout
        while time.time() < end:
            s = self.state()
            if s.get("state") is not None and pred(s):
                return s
            time.sleep(0.25)
        raise TimeoutError(f"timed out waiting for {what}")


def agent_dir(work):
    """A pi config directory with only pi-strata: the user's provider settings and credential, nothing else."""
    d = work / "pi-agent"
    d.mkdir(parents=True, exist_ok=True)
    user = json.loads((USER_AGENT_DIR / "settings.json").read_text())
    settings = {k: user[k] for k in ("lastChangelogVersion", "defaultProvider", "defaultModel", "theme") if k in user}
    settings.update(packages=[str(PI_STRATA)], hideThinkingBlock=True)
    (d / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    for name in ("strata.json", "models-store.json"):     # the endpoint, and the models pi has seen there
        shutil.copy(USER_AGENT_DIR / name, d / name)
    auth = d / "auth.json"
    if not auth.exists():
        auth.symlink_to(USER_AGENT_DIR / "auth.json")     # the credential stays in the user's file
    return d


def layout(sc, adir, work):
    """The tmux session: the caption on top, the agents side by side, the monitor at the bottom."""
    tmux("kill-session", "-t", DEMO, check=False)
    tmux("new-session", "-d", "-s", DEMO, "-x", str(COLS), "-y", str(ROWS))
    for opt, val in (("status", "off"), ("pane-border-status", "top"), ("pane-border-lines", "heavy"),
                     ("pane-border-format", " #{@label} "), ("pane-active-border-style", "fg=colour244"),
                     ("pane-border-style", "fg=colour238"), ("allow-rename", "off")):
        tmux("set-option", "-t", DEMO, opt, val)
    tmux("set-option", "-g", "default-terminal", "tmux-256color")
    tmux("set-option", "-s", "extended-keys", "on")
    tmux("set-option", "-s", "extended-keys-format", "csi-u")
    tmux("set-option", "-gp", "allow-set-title", "off")       # pi would retitle the panes "π - <dir>"
    cap = tmux("display-message", "-p", "-t", DEMO, "#{pane_id}").strip()
    text = work / f"{sc.name}.caption"
    text.write_text(f"\033[1m{sc.title}\033[0m\n" + "\n".join(textwrap.wrap(sc.caption, COLS - 2)))
    tmux("send-keys", "-t", cap, f"clear; stty -echo; cat {shlex.quote(str(text))}; cat > /dev/null", "Enter")
    tmux("set-option", "-p", "-t", cap, "@label", "Strata scheduler demo")
    mon = tmux("split-window", "-v", "-t", cap, "-l", str(ROWS - CAPTION_ROWS - 1), "-P", "-F", "#{pane_id}").strip()
    panes = {}
    agents_arg = ",".join(f"{n}={p}" for n, p in sc.agents)
    first = tmux("split-window", "-v", "-b", "-t", mon, "-l", str(ROWS - CAPTION_ROWS - MONITOR_ROWS - 3),
                 "-P", "-F", "#{pane_id}").strip()
    panes[sc.agents[0][0]] = first
    prev = first
    for i, (name, _) in enumerate(sc.agents[1:], 1):
        prev = tmux("split-window", "-h", "-t", prev, "-l", f"{100 // (len(sc.agents) - i + 1) * (len(sc.agents) - i)}%",
                    "-P", "-F", "#{pane_id}").strip()
        panes[name] = prev
    env = f"PI_CODING_AGENT_DIR={shlex.quote(str(adir))} PI_SKIP_VERSION_CHECK=1"
    tmux("set-option", "-p", "-t", mon, "@label", "Strata turn scheduler (/metrics)")
    pane_arg = ",".join(f"{n}={panes[n]}" for n, _ in sc.agents)
    tmux("send-keys", "-t", mon, f"clear; env {env} CONTAINER_CONNECTION={os.environ.get('CONTAINER_CONNECTION', '')} "
         f"python3 {shlex.quote(str(HERE / 'monitor.py'))} --agents {agents_arg} --panes {pane_arg} "
         f"--tmux-socket {SOCKET}", "Enter")
    for name, prio in sc.agents:
        cwd = work / f"{sc.name}-{name}"
        shutil.rmtree(cwd, ignore_errors=True)
        cwd.mkdir(parents=True)
        tmux("set-option", "-p", "-t", panes[name], "@label", f"Agent {name}  ·  priority {prio}")
        tmux("send-keys", "-t", panes[name], f"cd {shlex.quote(str(cwd))} && clear && env {env} "
             f"PI_STRATA_PRIORITY={prio} pi --no-session --thinking off --tools read,bash", "Enter")
    return panes


def say(panes, name, text):
    tmux("send-keys", "-t", panes[name], "-l", text)
    time.sleep(0.4)
    tmux("send-keys", "-t", panes[name], "Enter")


def settle(m, quiet_s=6.0, timeout=400):
    """Waits until the engine has been idle with nobody waiting for `quiet_s` seconds."""
    end, since = time.time() + timeout, None
    while time.time() < end:
        s = m.state()
        if s["state"] == "idle" and not s["waiting"]:
            since = since or time.time()
            if time.time() - since >= quiet_s:
                return
        else:
            since = None
        time.sleep(0.25)
    raise TimeoutError("the agents did not finish")


def play(name, panes, m, marks):
    """Drives the agents; `marks` gets the moments worth seeing in real time, as (time, seconds before, after)."""
    p = lambda a: PROMPTS[(name, a)]
    mark = lambda before, after: marks.append((time.time(), before, after))
    start = m.state()
    if name == "preempt":
        say(panes, "L", p("L"))
        mark(0.5, 2)
        m.wait("L to be writing", lambda s: s["state"] == "generating" and s["running"] == 10 and s["generated"] >= 250)
        time.sleep(2)
        say(panes, "H", p("H"))
        mark(1.5, 1)
        m.wait("the preemption", lambda s: s["preemptions"] > start["preemptions"])
        m.wait("H's answer", lambda s: s["done"] >= start["done"] + 1)
        mark(1, 3)                                      # the preemption, H's answer and L going on
        m.wait("L to finish", lambda s: s["done"] >= start["done"] + 2 and s["state"] == "idle", timeout=400)
        mark(1, 1)
    elif name == "yield":
        say(panes, "A", p("A"))
        mark(0.5, 2)
        m.wait("A's call", lambda s: s["done"] >= start["done"] + 1)            # its response with the call ended
        time.sleep(2)
        say(panes, "B", p("B"))
        mark(2, 1)
        m.wait("B to be writing", lambda s: s["state"] == "generating" and s["running"] == 10)
        mark(0, 3)                                      # B has the engine while A's command runs
        m.wait("A's follow-up", lambda s: s["done"] >= start["done"] + 2 or s["preemptions"] > start["preemptions"],
               timeout=120)
        mark(2, 4)
        m.wait("everyone to finish", lambda s: s["done"] >= start["done"] + 3, timeout=400)
        mark(1, 1)
    elif name == "same-priority":
        say(panes, "A", p("A"))
        mark(0.5, 1.5)
        m.wait("A to start", lambda s: s["state"] in ("reading", "generating"))
        time.sleep(1.5)
        say(panes, "B", p("B"))
        mark(0.5, 3)                                    # B waits while A goes on with its tool calls
        m.wait("A's turn to end", lambda s: s["done"] >= start["done"] + 5, timeout=300)
        mark(2, 3)                                      # A's summary, then B's request gets the engine
        m.wait("B to finish", lambda s: s["done"] >= start["done"] + 6, timeout=300)
        mark(1, 1)
    elif name == "reservation":
        say(panes, "A", p("A"))
        mark(0.5, 1.5)
        m.wait("A's call", lambda s: s["done"] >= start["done"] + 1)
        time.sleep(1.5)
        say(panes, "B", p("B"))
        mark(0.5, 3)                                    # B waits: the engine is reserved for A
        m.wait("A's follow-up", lambda s: s["done"] >= start["done"] + 2, timeout=300)
        mark(2.5, 1)
        m.wait("B to start", lambda s: s["running"] == 10 and s["state"] in ("reading", "generating"), timeout=300)
        mark(0.5, 3)
        m.wait("B to finish", lambda s: s["done"] >= start["done"] + 4, timeout=300)
        mark(1, 1)
    settle(m, quiet_s=2)
    time.sleep(1)


def record(sc, out, work, tools):
    adir = agent_dir(work)
    m = Metrics(adir)
    if m.state().get("state") is None:
        sys.exit("Strata's /metrics does not answer: is the deployment up?")
    m.wait("an idle engine", lambda s: s["state"] == "idle" and not s["waiting"], timeout=600)
    panes = layout(sc, adir, work)
    time.sleep(6)                                       # pi starts and discovers the model
    cast = work / f"{sc.name}.cast"
    tmux("kill-session", "-t", REC, check=False)
    tmux("new-session", "-d", "-s", REC, "-x", str(COLS), "-y", str(ROWS))
    tmux("set-option", "-t", REC, "status", "off")
    tmux("send-keys", "-t", REC, f"env -u TMUX {shlex.quote(tools['asciinema'])} rec --overwrite -q "
         f"--cols {COLS} --rows {ROWS} -c 'tmux -L {SOCKET} attach -t {DEMO}' {shlex.quote(str(cast))}", "Enter")
    time.sleep(3)
    marks = []
    try:
        play(sc.name, panes, m, marks)
        played = time.time()
    finally:
        tmux("kill-session", "-t", DEMO, check=False)   # the client exits, asciinema writes the cast
        time.sleep(2)
        tmux("kill-session", "-t", REC, check=False)
    (work / f"{sc.name}.marks.json").write_text(json.dumps(marks) + "\n")
    trim(cast, played)
    render(sc.name, out, work, tools)


def trim(cast, played):
    """Drops the end of the recording after the demo played (the session closing clears the screen)."""
    lines = cast.read_text().splitlines()
    end = played - json.loads(lines[0])["timestamp"]
    keep = [lines[0]]
    for x in lines[1:]:
        t, _, text = json.loads(x)
        if t > end or "\x1b[?1049l" in text:          # tmux leaving the screen as the session closes
            break
        keep.append(x)
    cast.write_text("\n".join(keep) + "\n")


def speed_up(cast, marks, fast):
    """Plays the marked moments in real time and the rest `fast` times faster (both sped up further if the video
    would still be too long), with a badge on the top border saying the speed.  Returns the fast-forwarded cast."""
    lines = cast.read_text().splitlines()
    head, events = json.loads(lines[0]), [json.loads(x) for x in lines[1:]]
    end = events[-1][0] if events else 0.0
    spans = []                                          # the real-time spans, merged
    for at, before, after in sorted(marks):
        a, b = max(0.0, at - head["timestamp"] - before), min(end, at - head["timestamp"] + after)
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        elif a < b:
            spans.append([a, b])
    real = sum(b - a for a, b in spans)
    slow = max(1.0, real / HIGHLIGHT_S)
    fast = max(fast, slow, (end - real) / max(1.0, MAX_S - LAST_FRAME_S - real / slow))
    cuts = [0.0] + [x for s in spans for x in s] + [end]
    rate = lambda t: slow if any(a <= t < b for a, b in spans) else fast

    def warp(t):                                        # recording time -> video time
        v = 0.0
        for a, b in zip(cuts, cuts[1:]):
            if t <= a:
                break
            v += (min(t, b) - a) / rate((a + b) / 2)
        return v

    def badge(r):
        text = "▶ real time" if r < 1.05 else f"▶▶ {r:.1f}×" if r < 3 else f"▶▶ {r:.0f}×"
        text = f" {text:>11} "                          # one width: a shorter badge covers a longer one
        style = "\033[1;37;42m" if r < 1.05 else "\033[1;30;43m"
        return f"\0337\033[1;{COLS - len(text) - 1}H{style}{text}\033[0m\0338"

    out, next_badge = [], 0.0
    for t, kind, text in events:
        v = warp(t)
        out.append([round(v, 4), kind, text])
        if v >= next_badge and not re.search(r"\x1b[^a-zA-Z]*$", text):   # not inside an escape sequence
            out.append([round(v, 4), "o", badge(rate(t))])  # tmux redraws the border now and then: draw it again
            next_badge = v + 0.25
    for a in cuts[1:-1]:                                # the badge changes as the speed does
        out.append([round(warp(a), 4), "o", badge(rate(a))])
    out.sort(key=lambda e: e[0])
    head.pop("idle_time_limit", None)
    fast_cast = cast.with_suffix(".fast.cast")
    fast_cast.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in [head] + out) + "\n")
    print(f"{cast.stem}: {real:.0f} s of {end:.0f} s at {slow:.1f}×, the rest at {fast:.1f}×: "
          f"{warp(end) + LAST_FRAME_S:.1f} s", flush=True)
    return fast_cast


def render(name, out, work, tools, fast=6.0):
    cast, gif = work / f"{name}.cast", work / f"{name}.gif"
    marks = work / f"{name}.marks.json"
    if marks.exists():
        cast = speed_up(cast, json.loads(marks.read_text()), fast)
    sh(tools["agg"], "--font-family", "Source Code Pro,Adwaita Mono", "--font-size", "16", "--idle-time-limit", "60",
       "--fps-cap", "30", "--last-frame-duration", str(LAST_FRAME_S), "--theme", "monokai", str(cast), str(gif))
    out.mkdir(parents=True, exist_ok=True)
    mp4 = out / f"{name}.mp4"
    sh(tools["ffmpeg"], "-y", "-loglevel", "error", "-i", str(gif), "-an", "-movflags", "+faststart",
       "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=15", "-c:v", "libx264", "-crf", "23",
       str(mp4))
    print(f"wrote {mp4}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--out", type=Path, default=HERE / "videos")
    ap.add_argument("--render-only", action="store_true", help="render the recordings already in --work")
    ap.add_argument("--work", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "strata-demo")
    args = ap.parse_args()
    tools = {k: os.environ.get(k.upper(), k) for k in ("asciinema", "agg", "ffmpeg")}
    for name in args.scenarios:
        if args.render_only:                            # an earlier recording: trim and render it again
            trim(args.work / f"{name}.cast", float("inf"))
            render(name, args.out, args.work, tools)
        else:
            record(SCENARIOS[name], args.out, args.work, tools)


if __name__ == "__main__":
    main()
