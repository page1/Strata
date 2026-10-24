"""My own tests for serve/leases.py (job lease-1-table): version numbers, the .bad rename, renew after expiry,
resume bookkeeping, checkpoint and the writer lock.  Temp folders only, no server, no GPU."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from serve import leases as L


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def table(d, clock, expire_s=45.0):
    return L.LeaseTable(Path(d), expire_s=expire_s, clock=clock)


class Versions(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.c = Clock()
        self.t = table(self.d, self.c)

    def tearDown(self):
        self.t.close()

    def test_version_goes_up_on_every_write(self):
        a = self.t.create(cmd="a", cwd=".", mode="full")
        f = Path(self.d) / f"{a['id']}.json"
        seen = [json.loads(f.read_text(encoding="utf-8"))["version"]]
        self.assertEqual(seen, [1])
        self.t.renew(a["id"])
        self.t.mark_woken(a["id"])
        self.t.end(a["id"], exit=0)
        after = json.loads(f.read_text(encoding="utf-8"))["version"]
        self.assertEqual(after, 4)                         # create + renew + mark_woken + end
        self.t.end(a["id"], exit=0)                        # idempotent: no write, no bump
        self.assertEqual(json.loads(f.read_text(encoding="utf-8"))["version"], 4)

    def test_resume_model_inherited_from_a_live_lease(self):
        a = self.t.create(cmd="a", cwd=".", mode="full", model_loaded=True)
        b = self.t.create(cmd="b", cwd=".", mode="full", model_loaded=False)
        self.assertTrue(b["resume_model"])
        self.t.end(a["id"], exit=0)
        c = self.t.create(cmd="c", cwd=".", mode="full", model_loaded=False)
        self.assertTrue(c["resume_model"])                 # b still holds, and b inherited it
        self.t.end(b["id"], exit=0)
        self.t.end(c["id"], exit=0)
        self.assertTrue(self.t.take_resume())

    def test_renew_after_expiry_raises_lease_ended(self):
        a = self.t.create(cmd="a", cwd=".", mode="full")
        self.c.t += 100
        self.assertEqual([x["id"] for x in self.t.expire_pass()], [a["id"]])
        with self.assertRaises(L.LeaseEnded):
            self.t.renew(a["id"])
        self.assertEqual(self.t.get(a["id"])["ended"], "expired")

    def test_renew_after_override_raises_lease_ended(self):
        a = self.t.create(cmd="a", cwd=".", mode="full")
        self.t.override_all()
        with self.assertRaises(L.LeaseEnded):
            self.t.renew(a["id"])

    def test_unknown_owner_bad_rename_and_grace(self):
        (Path(self.d) / ("ab" * 16 + ".json")).write_text("{not json", encoding="utf-8")
        self.t.close()
        self.t = table(self.d, self.c)
        self.assertFalse((Path(self.d) / ("ab" * 16 + ".json")).exists())
        self.assertTrue((Path(self.d) / ("ab" * 16 + ".json.bad")).exists())
        self.assertFalse(self.t.safe_to_load())
        self.assertEqual(self.t.live(), [])                # an unknown owner is not a lease
        self.c.t += 45
        self.assertEqual(self.t.expire_pass(), [])         # nothing to expire, only the grace ends
        self.assertTrue(self.t.safe_to_load())

    def test_override_all_returns_only_live(self):
        a = self.t.create(cmd="a", cwd=".", mode="full")
        b = self.t.create(cmd="b", cwd=".", mode="ram")
        self.t.end(b["id"], exit=0)
        self.assertEqual([x["id"] for x in self.t.override_all()], [a["id"]])
        self.assertEqual(self.t.override_all(), [])

    def test_checkpoint_missing_reads_none(self):
        self.assertIsNone(self.t.read_checkpoint())
        self.t.checkpoint({}, [])
        self.assertIsNotNone(self.t.read_checkpoint())

    def test_checkpoint_extra_keys_ride_in_the_same_write(self):
        self.t.checkpoint({"opencode": {"o1": "generating"}}, [], shutting_down=True)
        cp = self.t.read_checkpoint()
        self.assertTrue(cp["shutting_down"])
        self.assertEqual(cp["register"], {"opencode": {"o1": "generating"}})
        self.t.checkpoint({}, [])                        # a plain beat: no mark
        self.assertNotIn("shutting_down", self.t.read_checkpoint())

    def test_no_temp_files_left_behind(self):
        a = self.t.create(cmd="a", cwd=".", mode="full")
        self.t.renew(a["id"])
        self.t.end(a["id"], exit=0)
        self.t.checkpoint({"x": 1}, ["f.bin"])
        self.assertEqual(list(Path(self.d).glob("*.tmp")), [])

    def test_state_and_checkpoint_are_not_leases(self):
        self.t.create(cmd="a", cwd=".", mode="full")
        self.t.checkpoint({}, [])
        self.t.end(self.t.live()[0]["id"], exit=0)         # writes state.json too
        self.t.close()
        self.t = table(self.d, self.c)
        self.assertEqual(self.t.live(), [])
        self.assertTrue(self.t.safe_to_load())

    def test_pending_resume_moves_into_a_new_lease(self):
        a = self.t.create(cmd="a", cwd=".", mode="full", model_loaded=True)
        self.t.end(a["id"], exit=0)                        # resume pending, nobody took it yet
        self.assertTrue(json.loads((Path(self.d) / "state.json").read_text(encoding="utf-8"))["resume_pending"])
        b = self.t.create(cmd="b", cwd=".", mode="ram", model_loaded=False)
        self.assertTrue(b["resume_model"])                 # b's held period inherits the pending resume
        self.assertFalse(json.loads((Path(self.d) / "state.json").read_text(encoding="utf-8"))["resume_pending"])
        self.assertFalse(self.t.take_resume())             # not while b holds
        self.t.end(b["id"], exit=0)
        self.assertTrue(self.t.take_resume())
        self.assertFalse(self.t.take_resume())             # still only once

    def test_pending_resume_survives_a_restart_and_a_new_lease(self):
        a = self.t.create(cmd="a", cwd=".", mode="full", model_loaded=True)
        self.t.end(a["id"], exit=0)
        self.t.close()
        self.t = table(self.d, self.c)                     # the server restarted, resume still pending
        b = self.t.renew("cd" * 16)                        # a holder renews an id the table never saw
        self.assertTrue(b["resume_model"])
        self.assertFalse(self.t.take_resume())
        self.t.end(b["id"], exit=0)
        self.assertTrue(self.t.take_resume())

    def test_take_resume_false_while_a_lease_is_live(self):
        self.t.create(cmd="a", cwd=".", mode="full", model_loaded=True)
        self.t.create(cmd="b", cwd=".", mode="full", model_loaded=True)
        self.t.end(self.t.live()[0]["id"], exit=0)         # one still holds: no resume
        self.assertFalse(self.t.take_resume())
        self.assertTrue(self.t.live())                     # the live lease is untouched by the refused take


class WriterLock(unittest.TestCase):
    def test_second_table_in_this_process_is_busy(self):
        d = tempfile.mkdtemp()
        t = table(d, Clock())
        with self.assertRaises(L.LeaseTableBusy):
            table(d, Clock())
        t.close()
        t2 = table(d, Clock())                             # free again after close
        t2.close()

    def test_lock_and_unlock_cover_the_same_byte_with_a_non_empty_lock_file(self):
        d = tempfile.mkdtemp()
        (Path(d) / ".writer.lock").write_bytes(b"left over by an older version\n")
        for _ in range(3):                                 # open/close twice over a non-empty file: each open must
            t = table(d, Clock())                          # lock the same byte its close() unlocks, or the next
            with self.assertRaises(L.LeaseTableBusy):      # open would fail to see the lock (or this one would)
                table(d, Clock())
            t.close()
        self.assertEqual((Path(d) / ".writer.lock").read_bytes(), b"left over by an older version\n")
        t = table(d, Clock())                              # still free to open once the holder closed
        t.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
