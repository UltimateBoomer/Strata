"""serve/tune.py - "strata_tune": engine settings for one request, sent on its GEN line (no restart).

Two kinds of key.  The speed settings (pcie_frac, spec_min_p, prefill_pipe, aux_cpus) change how the work is placed,
not the answer: setup's calibration measures them, and the run config's sampling.strata_tune holds the value it chose.
The answer settings change which experts run or how much of a long prompt is read exactly, a trade of quality for
speed that a client may want to make:

- tail_skip (0-9; STRATA_ROUTE_TAIL_SKIP): a missed expert that every token of a verify window routes at this rank or
  lower is neither copied nor computed.  0 is off.
- kva (true/false; STRATA_KVA): the approximate late-layer read of a long cold prompt.
- route_prior (>= 0; STRATA_ROUTE_PRIOR): residency-biased routing, the logit bonus of an expert already in VRAM.
  0 is off.  Needs an engine started with STRATA_ROUTE_LIVE=1 (engine_env.route_live: 1).
- route_resident (>= 0; STRATA_ROUTE_RESIDENT) and route_resident_ranks ("lo-hi", 0-9; STRATA_ROUTE_RESIDENT_RANKS):
  a missed tail expert swapped for the best resident one within this many logits.  0 is off.  Needs STRATA_ROUTE_LIVE=1.

A request without a key runs with the engine's start-up value (the engine sets every key for every request: none
carries over).  The keys a request's own "strata_tune" may set are the sampling block's "client_tune" (a list); with
none, every key, as before.  The request's keys go over the run config's strata_tune one by one.
"""
from __future__ import annotations

import re

SPEED_KEYS = ("pcie_frac", "spec_min_p", "prefill_pipe", "aux_cpus")
ANSWER_KEYS = ("tail_skip", "kva", "route_prior", "route_resident", "route_resident_ranks")
KEYS = SPEED_KEYS + ANSWER_KEYS
_RANKS = re.compile(r"^\s*(\d)\s*-\s*(\d)\s*$")
ROUTE_MAX = 64.0        # a logit bonus or margin past this picks resident experts only; refused as a typo


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def value(key: str, v):
    """`v` as the engine takes it for `key`, or None when it is not a usable value."""
    if key in ("pcie_frac", "spec_min_p"):
        return float(v) if _number(v) and 0.0 <= float(v) <= 1.0 else None
    if key == "prefill_pipe":   # a layer split's prompt chunks: 0 off, 1 evened out, > 1 the rig's b/a in tokens
        return float(v) if _number(v) and 0.0 <= float(v) <= 262144.0 else None
    if key in ("aux_cpus", "kva"):   # true/false or 1/0
        return int(v) if isinstance(v, bool) or (_number(v) and v in (0, 1)) else None
    if key == "tail_skip":
        return int(v) if _number(v) and v == int(v) and 0 <= v <= 9 else None
    if key in ("route_prior", "route_resident"):
        return float(v) if _number(v) and 0.0 <= float(v) <= ROUTE_MAX else None
    if key == "route_resident_ranks":
        lo = hi = None
        if isinstance(v, str) and (m := _RANKS.match(v)):
            lo, hi = int(m.group(1)), int(m.group(2))
        elif isinstance(v, (list, tuple)) and len(v) == 2 and all(_number(x) and x == int(x) for x in v):
            lo, hi = int(v[0]), int(v[1])
        return f"{lo}-{hi}" if lo is not None and 0 <= lo <= hi <= 9 else None
    return None


def clean(tune, where: str) -> dict:
    """A config's strata_tune: its keys checked (ValueError naming the bad one), unknown keys refused."""
    if not isinstance(tune, dict):
        raise ValueError(f"{where}: expected an object of {', '.join(KEYS)}")
    out = {}
    for k, v in tune.items():
        if k not in KEYS:
            raise ValueError(f"{where}.{k}: not a tuning key (one of {', '.join(KEYS)})")
        if v is None:
            continue
        if value(k, v) is None:
            raise ValueError(f"{where}.{k}={v!r}: not a usable value")
        out[k] = v
    return out


def clean_client(keys, where: str) -> list[str]:
    """A config's client_tune: the strata_tune keys a request may set."""
    if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
        raise ValueError(f"{where}: expected a list of tuning keys")
    bad = [k for k in keys if k not in KEYS]
    if bad:
        raise ValueError(f"{where}: {', '.join(bad)} not tuning keys (one of {', '.join(KEYS)})")
    return list(dict.fromkeys(keys))


def merge(defaults, asked, client) -> tuple[dict | None, list[str]]:
    """The strata_tune a request runs with: the config's `defaults`, with the request's own `asked` keys over them one by
    one - only those in `client` when that is a list.  Returns it (None: nothing) and the keys of `asked` left out."""
    out = dict(defaults) if isinstance(defaults, dict) else {}
    dropped = []
    if isinstance(asked, dict):
        for k, v in asked.items():
            if isinstance(client, list) and k not in client:
                dropped.append(k)
            elif v is None:
                out.pop(k, None)
            else:
                out[k] = v
    return (out or None), dropped


def engine_keys(tune) -> str:
    """The GEN line's keys for a strata_tune (a value the engine cannot use is not sent)."""
    if not isinstance(tune, dict):
        return ""
    keys = ""
    for k in KEYS:
        v = value(k, tune.get(k)) if k in tune else None
        if v is None:
            continue
        name = "pipe" if k == "prefill_pipe" else k
        keys += f" {name}={v!r}" if isinstance(v, float) else f" {name}={v}"
    return keys
