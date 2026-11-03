# Scheduler demo videos

Short recordings of the fork's turn scheduler (`priority_*` in `config/server.json`; see the fork's
`docs/BATCHING.md`) with several [pi](https://pi.dev) agents on this deployment. They have no audio. Each video
shows one interaction and runs under 30 s: the moments that show the scheduler at work play in real time, and the
rest is fast-forwarded. A badge in the top right corner shows the speed (`▶ real time` or `▶▶ n×`). The agents are side by side, and below them the scheduler pane (`monitor.py`, polling
`/metrics`) shows:

- who holds the engine, and a timeline with one mark per second;
- the waiting requests by priority, any reservation, and the preemption counter;
- the server's preemption and completion lines.

| Video | What happens |
|---|---|
| `videos/preempt.mp4` | L (priority 10) writes a long story. H (priority 0) asks a quick question, which preempts L mid-reply. H answers, then L carries on from the token it stopped at. |
| `videos/yield.mp4` | A (priority 0) runs a 30 s command with a 300 s timeout. That counts as a long call, so A yields the engine and B (priority 10) writes meanwhile. When the command ends, A's follow-up preempts B. |
| `videos/same-priority.mp4` | A and B both have priority 0. A works through several quick tool calls and keeps the engine for its time slice and hold. B's request runs once A's turn ends. |
| `videos/reservation.mp4` | A (priority 0) runs a 10 s command with no timeout. That is a short call, so the engine stays reserved for A and B (priority 10) waits until A's turn ends. |

## Recording them again

With the deployment up:

```sh
python3 scheduler-demo/run.py [preempt|yield|same-priority|reservation ...] --work /some/tmp/dir
```

This needs `tmux`, `pi` with the `pi-strata` package (`~/Projects/pi-strata`), `asciinema`, `agg` and `ffmpeg`. Set
`ASCIINEMA`, `AGG` and `FFMPEG` to override where each command is found. `--render-only` renders the recordings
already in `--work` again.

Each agent runs `pi --no-session --thinking off --tools read,bash` in an empty directory, with a pi config directory
that loads only `pi-strata`, so no memory or web extensions affect the runs. The endpoint, model cache and credential
come from `~/.pi/agent`; the credential is symlinked, not copied. `PI_STRATA_PRIORITY` sets each agent's priority.
tmux runs on its own socket, so your own sessions are not touched.

The model's replies differ from run to run, so a run can come out differently, for example a story that ends before
the other agent's request arrives. Check each video before using it.
