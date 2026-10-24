"""strata_jobs.to_wake_from_leases: which sessions to wake from the lease folder (read only)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_jobs as J  # noqa: E402


def lease(d, lid, ended=None, session=None, woken=False, created=1.0):
    (Path(d) / f"{lid}.json").write_text(json.dumps({"id": lid, "ended": ended, "session": session,
                                                    "woken": woken, "created": created}), encoding="utf-8")


class ToWakeFromLeases(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_released_and_overridden_are_never_returned(self):
        lease(self.d, "a" * 32, ended="released", session={"agent": "claude", "id": "s1"})
        lease(self.d, "b" * 32, ended="overridden", session={"agent": "opencode", "id": "o1"})
        self.assertEqual(J.to_wake_from_leases(Path(self.d), clean_shutdown=False), [])

    def test_expired_lease_wins_over_in_flight_for_same_session(self):
        lease(self.d, "c" * 32, ended="expired", session={"agent": "opencode", "id": "o9"})
        (Path(self.d) / "checkpoint.json").write_text(json.dumps(
            {"register": {"opencode": {"o9": "generating", "o8": "x"}, "claude": "not a dict"}}), encoding="utf-8")
        out = J.to_wake_from_leases(Path(self.d), clean_shutdown=False)
        self.assertEqual(sorted((x["id"], x["lease"], x["cause"]) for x in out),
                         [("o8", None, "in-flight"), ("o9", "c" * 32, "expired")])

    def test_expired_lease_without_session_is_skipped(self):
        lease(self.d, "d" * 32, ended="expired", session=None)
        self.assertEqual(J.to_wake_from_leases(Path(self.d), clean_shutdown=True), [])


if __name__ == "__main__":
    unittest.main()
