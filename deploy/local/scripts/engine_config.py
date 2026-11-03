"""The checkout's tests' copy of the fork's serve/engine_fields.py: the run configs' "engine" fields as "args" and "env".
The server reads /config itself, fields included; nothing in the image uses this file.

A config names engine options as fields instead of an argv list: "engine" maps each option to its value
("conversation_cache_mib": 16384 is --conversation-cache-mib 16384; true is a bare flag, false or null leaves it out),
and "engine_env" maps each STRATA_ environment variable to its value ("load_image": "/data/load-image/q2" is
STRATA_LOAD_IMAGE; true is "1", false or null leaves it out). Raw "args" and "env" still work beside them.
"""


def _value(v):
    return str(v) if isinstance(v, (int, float, str)) else None


def expand(cfg: dict, where: str = 'config') -> dict:
    """`cfg` with "engine" appended to "args" and "engine_env" merged into "env". ValueError for a bad field."""
    cfg = dict(cfg)
    engine = cfg.pop('engine', None) or {}
    engine_env = cfg.pop('engine_env', None) or {}
    if not isinstance(engine, dict) or not isinstance(engine_env, dict):
        raise ValueError(f'{where}: "engine" and "engine_env" must be objects')
    args = list(cfg.get('args') or [])
    for key, v in engine.items():
        flag = '--' + key.replace('_', '-')
        if flag in args:
            raise ValueError(f'{where}: {flag} is set both in "args" and as engine.{key}')
        if v is True:
            args.append(flag)
        elif v is not False and v is not None:
            if _value(v) is None:
                raise ValueError(f'{where}: engine.{key} must be true, false, null, a number or a string, not {v!r}')
            args += [flag, _value(v)]
    env = dict(cfg.get('env') or {})
    for key, v in engine_env.items():
        name = 'STRATA_' + key.upper()
        if name in env:
            raise ValueError(f'{where}: {name} is set both in "env" and as engine_env.{key}')
        if v is True:
            env[name] = '1'
        elif v is not False and v is not None:
            if _value(v) is None:
                raise ValueError(f'{where}: engine_env.{key} must be true, false, null, a number or a string, not {v!r}')
            env[name] = _value(v)
    if args:
        cfg['args'] = args
    if env:
        cfg['env'] = env
    return cfg

