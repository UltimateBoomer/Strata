"""serve/parkfiles.py - the parked conversations on disk (--conversation-cache-dir), for the Monitor.

The engine parks each conversation in a file of its own, <dir>/park-<seq>.park; model variants that share a folder
each have <group>/<variant>/ under it (serve/models.py), and one --conversation-cache-mib budget covers them all.  A
file's size is what it takes on disk (it is sparse), as the engine's budget counts it.  The session the file belongs to
is the server's 64-bit tag of the session's name (serve/server.py session_tag), the directory's last field; the
names behind the tags are kept by SessionNames (a JSON file next to the log), so they survive a restart.  The folder's
`obsolete` file lists the files a "strata_drop" marked (evicted first).
"""
from __future__ import annotations

import json
import os
import re
import struct
import threading
from pathlib import Path

MAGIC = b"STRPARK\x01"
FILE = re.compile(r"^park-(\d+)\.park$")
MAX_FILES = 256


def _flag(args: list, flag: str) -> str | None:
    found = None
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            found = args[i + 1]
        elif isinstance(a, str) and a.startswith(flag + "="):
            found = a[len(flag) + 1:]
    return found


def folders(args: list) -> tuple[str | None, str | None]:
    """(the engine's own folder, the group's root or None) from its arguments."""
    return _flag(args, "--conversation-cache-dir"), _flag(args, "--conversation-cache-group")


def session_of(path: str | Path) -> int | None:
    """The session tag stored in a park file (0: none), None when the file cannot be read as one."""
    try:
        with open(path, "rb") as f:
            head = f.read(48)
            if len(head) < 48 or head[:8] != MAGIC:
                return None
            offset, length = struct.unpack_from("<QQ", head, 24)
            if length < 16:
                return None
            f.seek(offset + length - 8)
            tail = f.read(8)
            return struct.unpack("<Q", tail)[0] if len(tail) == 8 else None
    except OSError:
        return None


def _obsolete(folder: Path) -> set[int]:
    try:
        return {int(x) for x in (folder / "obsolete").read_text().split() if x.isdigit()}
    except OSError:
        return set()


def scan(args: list, names: dict | None = None) -> dict | None:
    """The parked files of the engine started with `args` and, with a group, of every variant beside it: {"dir",
    "group", "files": [{"variant", "file", "bytes", "at", "session", "name", "obsolete", "own"}]}, newest first; None
    without --conversation-cache-dir."""
    own, group = folders(args or [])
    if not own:
        return None
    names = names or {}
    root = Path(group or own)
    own_path = os.path.normpath(own)
    if not group:
        places = [(None, root)]
    else:
        try:
            places = [(d.name, Path(d.path)) for d in sorted(os.scandir(root), key=lambda d: d.name) if d.is_dir()]
        except OSError:
            places = []
    out = []
    for variant, folder in places:
        gone = _obsolete(folder)
        try:
            entries = list(os.scandir(folder))
        except OSError:
            continue
        for e in entries:
            m = FILE.match(e.name)
            if not m:
                continue
            try:
                st = e.stat()
            except OSError:
                continue
            tag = session_of(e.path)
            seq = int(m.group(1))
            out.append({"variant": variant, "file": seq, "bytes": st.st_blocks * 512,
                        "at": st.st_mtime, "session": f"{tag:016x}" if tag else None,
                        "name": names.get(tag) if tag else None, "obsolete": seq in gone,
                        "own": os.path.normpath(str(folder)) == own_path})
    out.sort(key=lambda f: f["at"], reverse=True)
    return {"dir": own, "group": group, "files": out[:MAX_FILES], "total_files": len(out),
            "total_bytes": sum(f["bytes"] for f in out)}


class SessionNames:
    """Session tag -> the name a request gave it (strata_session, X-Strata-Session, metadata.user_id), kept in a small
    JSON file so the Monitor can name the files an earlier run parked.  At most LIMIT names, the oldest dropped."""
    LIMIT = 2000

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.lock = threading.Lock()
        self.names: dict[int, str] = {}
        if self.path is not None:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self.names = {int(k, 16): str(v) for k, v in raw.items()}
            except (OSError, ValueError, AttributeError):
                pass

    def note(self, tag: int, name: str) -> None:
        with self.lock:
            if self.names.get(tag) == name:
                return
            self.names.pop(tag, None)
            self.names[tag] = name
            while len(self.names) > self.LIMIT:
                self.names.pop(next(iter(self.names)))
            snapshot = dict(self.names)
        if self.path is not None:
            try:
                tmp = self.path.with_name(self.path.name + ".tmp")
                tmp.write_text(json.dumps({f"{k:016x}": v for k, v in snapshot.items()}), encoding="utf-8")
                os.replace(tmp, self.path)
            except OSError:
                pass

    def snapshot(self) -> dict[int, str]:
        with self.lock:
            return dict(self.names)
