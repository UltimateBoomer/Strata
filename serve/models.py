"""serve/models.py - several variants of one model behind one server (the config's "models", opt-in).

A run config with "models" is the common part of every variant's config; each variant adds its own:

    {"models": {"q2": "server-q2.json", "q3": "server-q3.json"}, "default_model": "q2", "default_idle_s": 600,
     "args": [... the options every variant starts with ...], ...}

A variant is a file next to the config (or the object itself) with the keys of VARIANT_KEYS: its "model_name" and
"aliases" (what requests call it), "args" (over the common ones: a variant's option replaces the common one's,
strata_control/user_args.merge), "env" and "sampling" (key by key over the common ones) and "log".  Everything else -
the tokenizer, the template, the picture reader, the scheduler - is the server's, so the variants are one model in
other forms (the quantizations of one checkpoint); "parallel" and replicas are refused.

One engine runs at a time.  A request names its variant in "model" (a variant's key, model name or alias); a name no
variant has goes to the loaded one ("unknown_model": "current", the default), to the default one ("default") or is
refused ("error").  The turn scheduler (serve/turns.py) decides when the engine switches; the switch saves the
conversation in the engine to a session file ("model_session_dir", default: the log's folder) and restores it when that
variant is loaded again.  After "default_idle_s" (0: never) without a request the server switches back to
"default_model" (an unloaded engine is only pointed at it).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from strata_control import user_args

# the keys a variant may set (everything else is the common config's)
VARIANT_KEYS = frozenset({"model_name", "aliases", "args", "env", "sampling", "log", "max_context"})
# the common config's switching keys (not part of any variant's run config)
SWITCH_KEYS = frozenset({"models", "default_model", "default_idle_s", "model_max_wait_s", "model_min_resident_s",
                         "model_preempt", "unknown_model", "model_session_dir"})
UNKNOWN_MODES = ("current", "default", "error")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


class UnknownModel(ValueError):
    """A request named a model no variant has, with "unknown_model": "error"."""


@dataclass(eq=False)
class Variant:
    name: str
    cfg: dict                                       # its whole run config (the common one with its own keys over it)
    model_name: str = ""
    aliases: list = field(default_factory=list)
    spawn: tuple | None = None                      # the engine's start (StrataEngine.spawn), set by the server

    def names(self) -> list[str]:
        return [self.name, self.model_name, *self.aliases]


class Models:
    """The variants and the switching state the turn scheduler reads (active, since, committed and the policy)."""

    def __init__(self, variants: dict[str, Variant], default: str, active: str, *, idle_s: float = 0.0,
                 max_wait_s: float = 120.0, min_resident_s: float = 30.0, preempt: bool = False,
                 unknown: str = "current", session_dir: str | None = None, clock=time.monotonic):
        self.variants, self.default, self.active = variants, default, active
        self.idle_s, self.max_wait_s, self.min_resident_s = idle_s, max_wait_s, min_resident_s
        self.preempt, self.unknown, self.session_dir, self.clock = preempt, unknown, session_dir, clock
        self.since = clock()                        # when the active variant took the engine
        self.committed: str | None = None           # the variant the scheduler switches to next
        self.loaded = lambda: True                  # the engine is loaded (the server's Service.loaded)
        self.saved: dict[str, str] = {}             # a variant's conversation, saved when it was switched away
        self.switches = 0
        self.switch_s = 0.0                         # the switches' time, summed
        self.last_switch_s: float | None = None

    def resolve(self, asked) -> str | None:
        """The variant a request's "model" names; None: the loaded one.  UnknownModel with "unknown_model": "error"."""
        if isinstance(asked, str) and asked:
            for v in self.variants.values():
                if asked in v.names():
                    return v.name
        if self.unknown == "default":
            return self.default
        if self.unknown == "error" and asked:
            raise UnknownModel(f"model {asked!r} is not served here (one of: {', '.join(self.all_names())})")
        return None

    def all_names(self) -> list[str]:
        return [n for v in self.variants.values() for n in dict.fromkeys(v.names())]

    def switched(self, name: str, seconds: float) -> None:
        self.active, self.since, self.committed = name, self.clock(), None
        self.switches += 1
        self.switch_s += seconds
        self.last_switch_s = round(seconds, 1)

    def session_path(self, name: str) -> str:
        return str(Path(self.session_dir or ".") / f"variant-{name}.session")

    def view(self) -> dict:
        return {"active": self.active, "default": self.default, "committed": self.committed,
                "variants": {n: v.model_name for n, v in self.variants.items()},
                "saved": sorted(self.saved), "switches": self.switches, "switch_s": round(self.switch_s, 1),
                "last_switch_s": self.last_switch_s, "default_idle_s": self.idle_s}


def _number(cfg: dict, key: str, default: float) -> float:
    v = cfg.get(key, default)
    if v is None:
        v = default
    if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
        raise ValueError(f'"{key}" must be a number of seconds, not {v!r}')
    return float(v)


def variant_config(common: dict, own: dict, name: str) -> dict:
    """A variant's run config: the common one without the switching keys, with the variant's keys over it."""
    bad = sorted(set(own) - VARIANT_KEYS)
    if bad:
        raise ValueError(f'models.{name}: {", ".join(bad)} cannot differ between variants (a variant sets only '
                         f'{", ".join(sorted(VARIANT_KEYS))})')
    cfg = {k: v for k, v in common.items() if k not in SWITCH_KEYS}
    for k, v in own.items():
        if k == "args":
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                raise ValueError(f'models.{name}.args must be a list of strings')
            cfg["args"] = user_args.merge(list(common.get("args") or []), v)
        elif k in ("env", "sampling"):
            if not isinstance(v, dict):
                raise ValueError(f'models.{name}.{k} must be an object')
            cfg[k] = {**(common.get(k) or {}), **v}
        else:
            cfg[k] = v
    return cfg


def load(cfg: dict, config_path: str | None, startup: str | None = None) -> tuple[Models, dict]:
    """The config's "models" -> (Models, the run config of the variant the server starts with: `startup`, else the
    default).  ValueError for a config that cannot be used."""
    spec = cfg.get("models")
    if not isinstance(spec, dict) or not spec:
        raise ValueError('"models" must be an object of variant names to their config files (or objects)')
    base = Path(config_path).parent if config_path else Path(".")
    variants: dict[str, Variant] = {}
    for name, own in spec.items():
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise ValueError(f'models: {name!r} is not a usable variant name (letters, digits, ".", "_", "-")')
        if isinstance(own, str):
            path = base / own
            try:
                own = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError) as e:
                raise ValueError(f"models.{name}: cannot read {path}: {e}") from None
        if not isinstance(own, dict):
            raise ValueError(f"models.{name} must be a config file name or an object")
        vcfg = variant_config(cfg, own, name)
        aliases = vcfg.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [x.strip() for x in aliases.split(",") if x.strip()]
        if not isinstance(aliases, list) or not all(isinstance(x, str) for x in aliases):
            raise ValueError(f"models.{name}.aliases must be a list of names")
        variants[name] = Variant(name, vcfg, vcfg.get("model_name") or name, list(aliases))
    seen: dict[str, str] = {}
    for v in variants.values():
        for n in dict.fromkeys(v.names()):
            if seen.setdefault(n, v.name) != v.name:
                raise ValueError(f"models: the name {n!r} is both {seen[n]}'s and {v.name}'s")
    default = cfg.get("default_model") or next(iter(variants))
    if default not in variants:
        raise ValueError(f'"default_model" {default!r} is not one of the variants ({", ".join(variants)})')
    active = startup or default
    if active not in variants:
        raise ValueError(f'the variant to start with, {active!r}, is not one of {", ".join(variants)}')
    unknown = cfg.get("unknown_model", "current")
    if unknown not in UNKNOWN_MODES:
        raise ValueError(f'"unknown_model" must be one of {", ".join(UNKNOWN_MODES)}, not {unknown!r}')
    preempt = cfg.get("model_preempt", False)
    if not isinstance(preempt, bool):
        raise ValueError(f'"model_preempt" must be true or false, not {preempt!r}')
    for v in variants.values():
        if v.cfg.get("parallel", 1) not in (None, 1) or v.cfg.get("replicas") or "--batch" in v.cfg.get("args", []):
            raise ValueError("models: variants run one request at a time (no \"parallel\", --batch or replicas)")
    session_dir = cfg.get("model_session_dir") or (os.path.dirname(cfg["log"]) if cfg.get("log") else None)
    models = Models(variants, default, active, idle_s=_number(cfg, "default_idle_s", 0),
                    max_wait_s=_number(cfg, "model_max_wait_s", 120), min_resident_s=_number(cfg, "model_min_resident_s", 30),
                    preempt=preempt, unknown=unknown, session_dir=session_dir)
    return models, variants[active].cfg
