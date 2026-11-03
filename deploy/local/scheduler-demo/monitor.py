#!/usr/bin/env python3
"""The scheduler pane of the demo videos: polls Strata's /metrics and shows the engine's state, the turn scheduler's
priorities, a timeline of who held the engine each second, and the server's preemption lines.

    monitor.py --agents "A=10,B=0"     # agent name = its priority (the timeline names the running agent by it)

The endpoint and key come from STRATA_BASE_URL / STRATA_API_KEY (default: pi's strata.json and auth.json).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

RESET, DIM, BOLD = "\033[0m", "\033[2m", "\033[1m"
COLORS = ["\033[38;5;214m", "\033[38;5;45m", "\033[38;5;170m", "\033[38;5;112m"]


def endpoint():
    base = os.environ.get("STRATA_BASE_URL")
    agent_dir = Path(os.environ.get("PI_CODING_AGENT_DIR", Path.home() / ".pi" / "agent"))
    if not base:
        base = json.loads((agent_dir / "strata.json").read_text())["baseUrl"]
    key = os.environ.get("STRATA_API_KEY")
    if not key:
        key = json.loads((agent_dir / "auth.json").read_text())["strata"]["key"]
    return re.sub(r"/v1/?$", "", base.rstrip("/")), key


def fetch(base, key):
    req = urllib.request.Request(base + "/metrics", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=2) as r:
        return json.loads(r.read())


def follow_log(container, lines, lock):
    """The server's [strata] preempted / done lines, as they come (podman logs over the deployment's connection)."""
    cmd = ["podman", "logs", "-f", "--since", "1s", container]
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except OSError:
        return
    for line in p.stdout:
        line = line.strip()
        m = re.search(r"\[strata\] (preempted after \d+ tokens|done: \d+ tokens in \d+ s)", line)
        if m:
            text = m.group(1)
            if text.startswith("preempted"):
                text = "\033[31m" + text + ": it waits, then goes on from there\033[0m"
            with lock:
                lines.append((time.strftime("%H:%M:%S"), text))
                del lines[:-6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default="", help="name=priority,... (the agents of this demo)")
    ap.add_argument("--container", default="strata-local_strata-server_1")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--panes", default="", help="name=tmux pane id,...: tells agents of one priority apart")
    ap.add_argument("--tmux-socket", default=None)
    args = ap.parse_args()
    agents = [(n, int(p)) for n, p in (x.split("=") for x in args.agents.split(",") if x)]
    by_p = {}
    for i, (n, p) in enumerate(agents):
        by_p.setdefault(p, []).append((n, COLORS[i % len(COLORS)]))
    base, key = endpoint()
    log, lock = [], threading.Lock()
    threading.Thread(target=follow_log, args=(args.container, log, lock), daemon=True).start()
    timeline, last_tick = [], 0.0
    seen_preemptions, last_running, last_waiting, start = None, None, {}, None
    panes = dict(x.split("=") for x in args.panes.split(",") if x)
    shots, streaming, last_switches = {}, None, None

    def who_streams():
        """The agent whose transcript changed since the last look (its editor and footer left out: the spinner of
        a waiting agent moves too), or None when not exactly one did."""
        changed = []
        for name, pane in panes.items():
            cmd = ["tmux"] + (["-L", args.tmux_socket] if args.tmux_socket else []) + ["capture-pane", "-p", "-t", pane]
            try:
                text = subprocess.run(cmd, capture_output=True, text=True, timeout=2).stdout.splitlines()[:-6]
            except Exception:
                return None
            if shots.get(name) is not None and shots[name] != text:
                changed.append(name)
            shots[name] = text
        return changed[0] if len(changed) == 1 else None
    sys.stdout.write("\033[?25l")
    while True:
        try:
            m = fetch(base, key)
            err = None
        except Exception as e:                           # the server restarting, a proxy hiccup: keep the last view
            m, err = None, str(e)[:60]
        width = shutil.get_terminal_size((100, 12)).columns
        out = []
        if m is not None:
            live, turns = m.get("live", {}), m.get("turns", {})
            pr = turns.get("priority") or {}
            running = pr.get("running")
            if start is None:                            # the counters count from the start of this demo
                start = {"preemptions": pr.get("preemptions", 0), "switches": turns.get("switches", 0),
                        "reserve_idle_s": pr.get("reserve_idle_s", 0)}
            if live.get("state") in ("reading", "generating") and running is not None:
                who = by_p.get(running)
                if who and len(who) > 1 and panes:          # one priority, several agents: the one streaming
                    if turns.get("switches") != last_switches:
                        streaming = None                    # another conversation took the engine
                    streaming = who_streams() or streaming
                    pick = [x for x in who if x[0] == streaming]
                    who = pick or who
                name = "/".join(n for n, _ in who) if who else f"p{running}"
                color = who[0][1] if who and len(who) == 1 else BOLD
                what = (f"reading the prompt" if live["state"] == "reading" else
                        f"generating  {live.get('generated') or 0} tok  {live.get('tok_s') or 0:.0f} tok/s")
                engine = f"{color}{BOLD}{name}{RESET} (priority {running})  {what}"
                mark = (color, name[0] if who and len(who) == 1 else "#")
            else:
                engine, mark = f"{DIM}idle{RESET}", (DIM, "·")
            if time.time() - last_tick >= 1.0:
                last_tick = time.time()
                timeline.append(mark)
                del timeline[:-(width - 12)]
            waiting = pr.get("waiting") or {}
            n = pr.get("preemptions", 0)
            if seen_preemptions is not None and n > seen_preemptions and last_running is not None:
                def label(p):
                    who = by_p.get(p)
                    return f"{who[0][0]} (p{p})" if who and len(who) == 1 else f"p{p}"
                better = min([int(k) for k in list(waiting) + list(last_waiting)] +
                             ([running] if running is not None and running != last_running else []),
                             default=running)
                with lock:
                    log.append((time.strftime("%H:%M:%S"), f"\033[1;31mPREEMPTED\033[0m {label(last_running)} "
                                f"stopped for {label(better)}"))
            seen_preemptions, last_waiting, last_switches = n, waiting, turns.get("switches")
            if running is not None:
                last_running = running
            wait_txt = ", ".join(f"{v} at priority {k}" for k, v in waiting.items()) or "none"
            res = pr.get("reserved_for")
            out.append(f"{BOLD}Strata turn scheduler{RESET}  {DIM}preempt={pr.get('preempt')} "
                       f"delta={pr.get('delta')} reserve={pr.get('reserve_s'):g}s aging={pr.get('aging_s'):g}s"
                       f"{RESET}")
            out.append(f"engine    {engine}")
            out.append(f"waiting   {wait_txt}")
            out.append(f"reserved  {'for priority ' + str(res) + ' (its tool call)' if res is not None else '-'}"
                       f"{'   holding for a follow-up' if turns.get('holding') else ''}")
            out.append(f"counters  preemptions {pr.get('preemptions', 0) - start['preemptions']}   "
                       f"switches {turns.get('switches', 0) - start['switches']}   "
                       f"reserve idle {pr.get('reserve_idle_s', 0) - start['reserve_idle_s']:.0f}s")
        else:
            out.append(f"{BOLD}Strata turn scheduler{RESET}  {DIM}(waiting for /metrics: {err}){RESET}")
            out += [""] * 4
        legend = "  ".join(f"{c}{n[0]}{RESET}={n} (p{p})" for p, xs in by_p.items() for n, c in xs)
        out.append(f"timeline  {''.join(c + ch + RESET for c, ch in timeline)}")
        out.append(f"{DIM}          one mark per second: {RESET}{legend}  {DIM}·=idle{RESET}")
        with lock:
            for at, text in log[-4:]:
                out.append(f"{DIM}log {at}{RESET} {text}")
        sys.stdout.write("\033[H\033[J" + "\n".join(line for line in out) + "\n")
        sys.stdout.flush()
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stdout.write("\033[?25h")
