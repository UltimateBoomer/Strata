"""serve/engine_fields.py - engine options as fields of the run config instead of an argv list (opt-in).

"engine" maps each engine option to its value ("conversation_cache_mib": 16384 is --conversation-cache-mib 16384; true
is a bare flag, false or null leaves it out), and "engine_env" maps each STRATA_ environment variable to its value
("load_image": "/data/load-image/q2" is STRATA_LOAD_IMAGE; true is "1", false or null leaves it out).  They are appended
to "args" and merged into "env" as the config is read, in the main config and in each model variant's file
(serve/models.py), so a variant's field replaces the common one's like any other option.  Raw "args" and "env" still
work beside them; one option set both ways is refused.
"""
from __future__ import annotations


def _value(v):
    return str(v) if isinstance(v, (int, float, str)) and not isinstance(v, bool) else None


def expand(cfg: dict, where: str = "config") -> dict:
    """`cfg` with "engine" appended to "args" and "engine_env" merged into "env" (a copy).  ValueError for a bad field."""
    if not isinstance(cfg, dict) or ("engine" not in cfg and "engine_env" not in cfg):
        return cfg
    cfg = dict(cfg)
    engine = cfg.pop("engine", None) or {}
    engine_env = cfg.pop("engine_env", None) or {}
    if not isinstance(engine, dict) or not isinstance(engine_env, dict):
        raise ValueError(f'{where}: "engine" and "engine_env" must be objects')
    args = list(cfg.get("args") or [])
    for key, v in engine.items():
        flag = "--" + key.replace("_", "-")
        if flag in args:
            raise ValueError(f'{where}: {flag} is set both in "args" and as engine.{key}')
        if v is True:
            args.append(flag)
        elif v is not False and v is not None:
            if _value(v) is None:
                raise ValueError(f"{where}: engine.{key} must be true, false, null, a number or a string, not {v!r}")
            args += [flag, _value(v)]
    env = dict(cfg.get("env") or {})
    for key, v in engine_env.items():
        name = "STRATA_" + key.upper()
        if name in env:
            raise ValueError(f'{where}: {name} is set both in "env" and as engine_env.{key}')
        if v is True:
            env[name] = "1"
        elif v is not False and v is not None:
            if _value(v) is None:
                raise ValueError(f"{where}: engine_env.{key} must be true, false, null, a number or a string, not {v!r}")
            env[name] = _value(v)
    if args:
        cfg["args"] = args
    if env:
        cfg["env"] = env
    return cfg
