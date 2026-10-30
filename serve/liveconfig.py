"""serve/liveconfig.py - the web page's Config view: the run config's files, edited and applied while the server runs.

The files are the main run config (--config) and, with model variants, every file its "models" names in the same folder
(serve/models.py).  GET /api/config hands them out as they are on disk; POST /api/config takes them back whole
({"files": {name: object}}), checks them the way a start would (the JSON, the "engine" fields, the variants, the
sampling, the scheduler's keys) and, unless "dry_run", writes them (the earlier ones kept in a history folder) and
applies them: Service.apply_config.

What applies without a restart of the server:
- the engine's start (exe, cwd, args, env, "engine", "engine_env", log, max_context, mtp): the engine is restarted with
  it when the loaded variant's start changed (another variant's is used at its next switch); a start that fails puts the
  earlier files back and starts the engine as it was;
- "sampling", "model_name", "aliases", the variants and their switching keys (serve/models.py SWITCH_KEYS), the turn
  scheduler's keys (serve/turns.py), "park" and "reasoning_budget_tokens": at once.
Every other key that changed (the tokenizer, the picture reader, the network and security keys, ...) is saved and named
in the answer's "restart_required": it is used from the next start of the server.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

from serve import engine_fields
from serve import models as serve_models

FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\.json")
ENGINE_KEYS = frozenset({"exe", "cwd", "args", "env", "engine", "engine_env", "log", "max_context", "mtp"})
TURN_KEYS = frozenset({"turn_slice_s", "turn_max_wait_s", "turn_hold_s", "turn_hold_skip_timeout_s", "turn_side_wait_s",
                       "priority", "priority_preempt", "priority_preempt_delta", "priority_preempt_timeout_s",
                       "priority_min_run_s", "priority_reserve_s", "priority_aging_s", "priority_default",
                       "priority_floor"})
LIVE_KEYS = frozenset({"sampling", "model_name", "aliases", "park", "reasoning_budget_tokens"}) | TURN_KEYS | \
    serve_models.SWITCH_KEYS
HISTORY_KEEP = 50                                   # applied versions kept in the history folder


def variant_files(main: dict) -> dict[str, str]:
    """The variant name -> file name of each variant the main config keeps in a file of its own."""
    spec = main.get("models") if isinstance(main, dict) else None
    if not isinstance(spec, dict):
        return {}
    return {k: v for k, v in spec.items() if isinstance(v, str)}


def read_files(main_path: str | Path) -> dict[str, dict]:
    """The main config and its variants' files, by file name (the main one first).  ValueError for one that cannot be
    read."""
    main_path = Path(main_path)
    out = {main_path.name: _read(main_path)}
    for name in variant_files(out[main_path.name]).values():
        if not FILE_NAME.fullmatch(name):
            raise ValueError(f"models: {name!r} is not a file next to {main_path.name}")
        out[name] = _read(main_path.parent / name)
    return out


def _read(path: Path) -> dict:
    try:
        cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as e:
        raise ValueError(f"{path.name}: cannot read it ({e.strerror or e})") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"{path.name}: not valid JSON ({e.msg} at line {e.lineno}, column {e.colno})") from None
    if not isinstance(cfg, dict):
        raise ValueError(f"{path.name}: not a JSON object")
    return cfg


def check_files(main_name: str, files: dict) -> dict[str, dict]:
    """The files a POST sends, checked for their shape: the main one is there, every name is a plain file name, every
    value an object, and every variant file the main one names is sent (a file it no longer names is left alone)."""
    if not isinstance(files, dict) or main_name not in files:
        raise ValueError(f'send {{"files": {{"{main_name}": {{...}}, ...}}}}')
    for name, cfg in files.items():
        if not isinstance(name, str) or not FILE_NAME.fullmatch(name):
            raise ValueError(f"{name!r} is not a config file name (letters, digits, '.', '_', '-', ending .json)")
        if not isinstance(cfg, dict):
            raise ValueError(f"{name}: expected a JSON object")
    wanted = set(variant_files(files[main_name]).values())
    missing = sorted(wanted - set(files))
    if missing:
        raise ValueError(f"{main_name} names {', '.join(missing)}, which was not sent")
    extra = sorted(set(files) - wanted - {main_name})
    if extra:
        raise ValueError(f"{', '.join(extra)}: not named in {main_name}'s \"models\"")
    return files


def changed_keys(old: dict, new: dict) -> list[str]:
    return sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))


def classify(main_name: str, old: dict[str, dict], new: dict[str, dict]) -> dict:
    """What changed, file by file and key by key, and which of those keys need the server restarted."""
    changes, restart = {}, []
    for name in sorted(set(old) | set(new)):
        keys = changed_keys(old.get(name) or {}, new.get(name) or {})
        if name not in new or not keys:
            continue
        changes[name] = keys
        if name == main_name:                           # (a variant's file sets only VARIANT_KEYS: all of them apply)
            restart += [k for k in keys if k not in LIVE_KEYS and k not in ENGINE_KEYS]
    return {"changes": changes, "restart_required": sorted(set(restart))}


def write_temp(folder: Path, files: dict[str, dict]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, cfg in files.items():
        (folder / name).write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return folder


def build(main_path: Path, startup: str | None, prepare=None) -> tuple[dict, serve_models.Models | None]:
    """The run config a start would use from the files at `main_path` (and, with variants, the Models, starting with
    `startup` when it is still one of them).  `prepare` (cfg -> cfg) is what the server does to a config before its
    variants are read (the saved tuning, the joined args).  ValueError for a config that cannot be used."""
    cfg = engine_fields.expand(_read(main_path), main_path.name)
    if prepare is not None:
        cfg = prepare(cfg)
    if cfg.get("models") is None:
        return cfg, None
    spec = cfg.get("models")
    if startup is not None and isinstance(spec, dict) and startup not in spec:
        startup = None
    models, cfg = serve_models.load(cfg, str(main_path), startup)
    return cfg, models


def save(main_path: Path, files: dict[str, dict], history: Path) -> Path:
    """`files` written over the folder's (each through a temporary file), the earlier ones first copied to a folder of
    the history, which is returned.  The oldest versions beyond HISTORY_KEEP go."""
    folder = main_path.parent
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = history / stamp
    n = 1
    while dest.exists():
        n += 1
        dest = history / f"{stamp}-{n}"
    dest.mkdir(parents=True)
    for name in files:
        if (folder / name).is_file():
            shutil.copy2(folder / name, dest / name)
    for name, cfg in files.items():
        tmp = folder / f".{name}.tmp"
        tmp.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        tmp.replace(folder / name)
    old = sorted((p for p in history.iterdir() if p.is_dir()), key=lambda p: p.name)
    for p in old[:-HISTORY_KEEP]:
        shutil.rmtree(p, ignore_errors=True)
    return dest


def restore(main_path: Path, backup: Path, names) -> None:
    """The files `names` put back from a history folder (a file that did not exist before is removed)."""
    folder = main_path.parent
    for name in names:
        src = backup / name
        if src.is_file():
            shutil.copy2(src, folder / name)
        else:
            (folder / name).unlink(missing_ok=True)
