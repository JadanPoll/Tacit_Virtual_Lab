"""Run: python3 gdb/test_incidents.py. Tests exercise native Linux processes."""
from __future__ import annotations

import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent

def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

class Incidents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="tacit-gdb-")
        cls.root = Path(cls.tmp.name)
        cls.executable = cls.root / "linux_incidents"
        subprocess.run([
            "cc", "-std=c11", "-g3", "-O0", "-fno-omit-frame-pointer",
            "-Wall", "-Wextra", "-Wpedantic", "-Werror", "-pthread",
            str(HERE / "linux_incidents.c"), "-o", str(cls.executable)
        ], check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_case(self, name, timeout=4):
        return subprocess.run([str(self.executable), name], cwd=self.root,
                              text=True, capture_output=True, timeout=timeout,
                              preexec_fn=no_core)

    def test_frame_total_is_reproducibly_wrong(self):
        p = self.run_case("frame")
        self.assertEqual(p.returncode, 0)
        self.assertIn("Report total: 123 (expected 26)", p.stdout)

    def test_watchpoint_target_changes(self):
        p = self.run_case("watch")
        self.assertEqual(p.returncode, 0)
        self.assertIn("max_requests: 3 (expected 32)", p.stdout)

    def test_abort_is_real_signal(self):
        p = self.run_case("abort")
        self.assertEqual(p.returncode, -signal.SIGABRT)
        self.assertIn("declared payload length 7 exceeds received 5", p.stderr)

    def test_thread_case_stays_blocked(self):
        with self.assertRaises(subprocess.TimeoutExpired) as ctx:
            self.run_case("threads", timeout=1.2)
        self.assertIn(b"Two workers started", ctx.exception.stdout or b"")

    def test_syscall_boundary_missing_file(self):
        p = self.run_case("syscall")
        self.assertEqual(p.returncode, 0)
        self.assertIn("Service startup: failed", p.stdout)
        self.assertIn("errno=2", p.stderr)

    def gdb(self, case, *commands):
        if not shutil.which("gdb"):
            if os.environ.get("GDB_REQUIRED") == "1":
                self.fail("GDB_REQUIRED=1 but gdb is unavailable")
            self.skipTest("GDB is not installed here")
        args = ["gdb", "-q", "-nx", "--batch", "-ex", "set pagination off"]
        for command in commands:
            args += ["-ex", command]
        args += ["--args", str(self.executable), case]
        return subprocess.run(args, cwd=self.root, text=True, capture_output=True, timeout=12)

    def test_gdb_stack_and_symbols(self):
        p = self.gdb("frame", "break sum_records", "run", "bt 4", "print record_count")
        self.assertIn("Breakpoint", p.stdout)
        self.assertIn("sum_records", p.stdout)
        self.assertIn("prepare_report", p.stdout)
        self.assertIn("= 4", p.stdout)

    def test_gdb_data_watchpoint(self):
        p = self.gdb("watch", "break accept_packet", "run", "disable 1",
                     "watch session_state.max_requests", "continue", "print session_state.max_requests")
        self.assertIn("watchpoint", p.stdout.lower())
        self.assertIn("New value = 3", p.stdout)

    def test_gdb_abort_backtrace(self):
        p = self.gdb("abort", "run", "bt 10")
        self.assertIn("SIGABRT", p.stdout)
        self.assertIn("enforce_message_contract", p.stdout)
        self.assertIn("decode_message", p.stdout)

if __name__ == "__main__":
    unittest.main(verbosity=2)
