"""serve/test_parkfiles.py - the Monitor's list of parked conversation files, and the scheduler's queue view.

    python -m unittest serve.test_parkfiles -v
"""
from __future__ import annotations

import struct
import tempfile
import threading
import unittest
from pathlib import Path

from serve import parkfiles
from serve.turns import Priorities, Turn, TurnScheduler


def park_file(path: Path, session: int, directory: bytes = b"PARKDIR1" + b"\0" * 64) -> None:
    """A file shaped like the engine's: a superblock naming the directory, the session tag its last field."""
    body = directory + struct.pack("<Q", session)
    head = parkfiles.MAGIC + struct.pack("<II", 1, 4096) + struct.pack("<QQQ", 1, 4096, len(body))
    path.write_bytes(head.ljust(4096, b"\0") + body)


class ScanTests(unittest.TestCase):
    def test_group_lists_every_variant_with_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for v in ("q2", "q3"):
                (root / v).mkdir()
            park_file(root / "q2" / "park-3.park", 0xabc)
            park_file(root / "q3" / "park-7.park", 0)
            (root / "q3" / "obsolete").write_text("7\n")
            (root / "q3" / "identity").write_text("x")
            got = parkfiles.scan(["--conversation-cache-dir", str(root / "q2"), "--conversation-cache-group", tmp],
                                 {0xabc: "agent-1"})
        files = {f["file"]: f for f in got["files"]}
        self.assertEqual(set(files), {3, 7})
        self.assertEqual((files[3]["variant"], files[3]["session"], files[3]["name"], files[3]["own"]),
                         ("q2", f"{0xabc:016x}", "agent-1", True))
        self.assertEqual((files[7]["variant"], files[7]["session"], files[7]["obsolete"], files[7]["own"]),
                         ("q3", None, True, False))
        self.assertEqual(got["total_files"], 2)

    def test_single_folder_and_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            park_file(Path(tmp) / "park-1.park", 5)
            (Path(tmp) / "park-2.park").write_bytes(b"not a park file")
            got = parkfiles.scan([f"--conversation-cache-dir={tmp}"])
        self.assertEqual([(f["file"], f["variant"]) for f in sorted(got["files"], key=lambda f: f["file"])],
                         [(1, None), (2, None)])
        self.assertEqual({f["file"]: f["session"] for f in got["files"]}, {1: f"{5:016x}", 2: None})
        self.assertIsNone(parkfiles.scan(["--kv", "int8"]))

    def test_names_survive_a_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session-names.json"
            parkfiles.SessionNames(path).note(42, "main")
            self.assertEqual(parkfiles.SessionNames(path).snapshot(), {42: "main"})


class QueueViewTests(unittest.TestCase):
    def test_running_and_waiting(self):
        now = [100.0]
        s = TurnScheduler(threading.Lock(), slice_s=60, clock=lambda: now[0], prio=Priorities())
        a = s.enter(Turn(session_key="A", priority=0))
        self.assertTrue(s.acquire(a, 1))
        s.enter(Turn(session_key="B", priority=-5, model="q3"))
        now[0] = 104.0
        v = s.view()
        self.assertEqual(v["running"]["session"], "A")
        self.assertEqual(v["running"]["held_s"], 4.0)
        self.assertEqual([(q["session"], q["priority"], q["model"], q["waited_s"]) for q in v["queue"]],
                         [("B", -5, "q3", 4.0)])


if __name__ == "__main__":
    unittest.main()
