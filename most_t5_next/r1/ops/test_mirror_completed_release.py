"""Linux-only race tests for mirror_completed_release.sh using fake transports."""

from __future__ import print_function

import os
import shutil
import subprocess
import tempfile
import textwrap
import time
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().with_name("mirror_completed_release.sh")


class MirrorCompletedReleaseTest(unittest.TestCase):
    def setUp(self):
        if os.name != "posix":
            self.skipTest("mirror race harness requires POSIX process semantics")
        self.bash = shutil.which("bash")
        self.sha256sum = shutil.which("sha256sum")
        if not self.bash or not self.sha256sum:
            self.skipTest("bash and sha256sum are required")
        self.root = Path(tempfile.mkdtemp()).resolve()
        self.bin = self.root / "bin"
        self.source = self.root / "source"
        self.destination = self.root / "destination"
        self.shard = self.source / "shard-000000"
        for directory in (self.bin, self.source, self.destination, self.shard):
            directory.mkdir()
        self.identity = self.root / "identity"
        self.identity.write_text("fixture\n", encoding="utf-8")
        self.shard_data = self.shard / "data"
        self.shard_data.write_text("complete shard\n", encoding="utf-8")
        self.manifest = self.source / "full_release_manifest.json"
        self.destination_manifest = self.destination / "full_release_manifest.json"
        self.state = self.root / "rsync-count"
        self.ssh_state = self.root / "ssh-count"
        self.ready = self.root / "ready"
        self.continue_file = self.root / "continue"
        self.destination_shard_data = self.destination / "shard-data"
        self.fake_rsync = self.bin / "rsync"
        self.fake_ssh = self.bin / "ssh"
        self._write_executable(
            self.fake_rsync,
            """
            #!/bin/sh
            set -eu
            count=0
            if [ -f "$TEST_STATE" ]; then count=$(cat "$TEST_STATE"); fi
            count=$((count + 1))
            printf '%s\n' "$count" > "$TEST_STATE"
            if [ "${TEST_MODE:-normal}:$count" = "late:1" ]; then
              : > "$TEST_READY"
              while [ ! -f "$TEST_CONTINUE" ]; do sleep 0.01; done
              cp "$TEST_SOURCE/shard-000000/data" "$TEST_DEST/shard-data"
            elif [ "${TEST_MODE:-normal}:$count" = "mutate:1" ]; then
              : > "$TEST_READY"
              while [ ! -f "$TEST_CONTINUE" ]; do sleep 0.01; done
              cp "$TEST_SOURCE/full_release_manifest.json" "$TEST_DEST/full_release_manifest.json"
            else
              cp "$TEST_SOURCE/full_release_manifest.json" "$TEST_DEST/full_release_manifest.json"
            fi
            """,
        )
        self._write_executable(
            self.fake_ssh,
            """
            #!/bin/sh
            set -eu
            count=0
            if [ -f "$TEST_SSH_STATE" ]; then count=$(cat "$TEST_SSH_STATE"); fi
            count=$((count + 1))
            printf '%s\n' "$count" > "$TEST_SSH_STATE"
            if [ "${TEST_REMOTE_MISMATCH_ONCE:-0}" = 1 ] && [ "$count" -eq 1 ]; then
              printf '%064d  %s\n' 0 "$TEST_DEST/full_release_manifest.json"
            else
              "$REAL_SHA256SUM" -- "$TEST_DEST/full_release_manifest.json"
            fi
            """,
        )

    def tearDown(self):
        if not hasattr(self, "root"):
            return
        for path in (
            self.destination_manifest,
            self.destination_shard_data,
            self.manifest,
            self.shard_data,
            self.fake_rsync,
            self.fake_ssh,
            self.identity,
            self.state,
            self.ssh_state,
            self.ready,
            self.continue_file,
        ):
            if path.exists() or path.is_symlink():
                path.unlink()
        for directory in (self.shard, self.source, self.destination, self.bin, self.root):
            if directory.exists():
                directory.rmdir()

    def _write_executable(self, path, body):
        path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
        path.chmod(0o755)

    def _environment(self, mode="normal", mismatch_once=False):
        environment = dict(os.environ)
        environment.update(
            {
                "PATH": str(self.bin) + os.pathsep + environment.get("PATH", ""),
                "TEST_MODE": mode,
                "TEST_STATE": str(self.state),
                "TEST_SSH_STATE": str(self.ssh_state),
                "TEST_READY": str(self.ready),
                "TEST_CONTINUE": str(self.continue_file),
                "TEST_SOURCE": str(self.source),
                "TEST_DEST": str(self.destination),
                "REAL_SHA256SUM": self.sha256sum,
                "TEST_REMOTE_MISMATCH_ONCE": "1" if mismatch_once else "0",
            }
        )
        return environment

    def _start(self, mode="normal", mismatch_once=False):
        return subprocess.Popen(
            [
                self.bash,
                str(SCRIPT),
                str(self.source),
                "fixture.invalid",
                "22",
                str(self.identity),
                "/retained/release",
                "0",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._environment(mode, mismatch_once),
        )

    def _wait_ready(self, process):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if self.ready.exists():
                return
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                self.fail("mirror exited before fake rsync was released: {} {}".format(stdout, stderr))
            time.sleep(0.01)
        process.kill()
        process.communicate()
        self.fail("timed out waiting for fake rsync file-list barrier")

    def _finish(self, process):
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            self.fail("mirror did not terminate: {} {}".format(stdout, stderr))
        self.assertEqual(process.returncode, 0, stdout + stderr)
        return stdout, stderr

    def test_manifest_created_after_first_file_list_requires_second_pass(self):
        process = self._start(mode="late")
        self._wait_ready(process)
        self.manifest.write_text('{"release_status":"complete"}\n', encoding="utf-8")
        self.continue_file.write_text("continue\n", encoding="utf-8")
        self._finish(process)
        self.assertEqual(self.state.read_text(encoding="utf-8").strip(), "2")
        self.assertEqual(self.destination_manifest.read_bytes(), self.manifest.read_bytes())

    def test_source_hash_change_during_rsync_forces_another_pass(self):
        self.manifest.write_text('{"generation":1}\n', encoding="utf-8")
        process = self._start(mode="mutate")
        self._wait_ready(process)
        self.manifest.write_text('{"generation":2}\n', encoding="utf-8")
        self.continue_file.write_text("continue\n", encoding="utf-8")
        self._finish(process)
        self.assertEqual(self.state.read_text(encoding="utf-8").strip(), "2")
        self.assertEqual(self.ssh_state.read_text(encoding="utf-8").strip(), "1")
        self.assertEqual(self.destination_manifest.read_bytes(), self.manifest.read_bytes())

    def test_destination_hash_mismatch_forces_another_pass(self):
        self.manifest.write_text('{"release_status":"complete"}\n', encoding="utf-8")
        process = self._start(mismatch_once=True)
        self._finish(process)
        self.assertEqual(self.state.read_text(encoding="utf-8").strip(), "2")
        self.assertEqual(self.ssh_state.read_text(encoding="utf-8").strip(), "2")
        self.assertEqual(self.destination_manifest.read_bytes(), self.manifest.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
