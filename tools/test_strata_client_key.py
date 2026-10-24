"""tools/test_strata_client_key.py - local: where tools/strata_suspend.py finds the key for its own requests
(STRATA_API_KEY, then this account's %APPDATA%\\Strata\\api-key, then the run config). No server is started."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_suspend as S  # noqa: E402


class ClientKey(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(os.environ, {"APPDATA": self.d.name})
        self.env.start()
        os.environ.pop("STRATA_API_KEY", None)

    def tearDown(self):
        self.env.stop()
        self.d.cleanup()

    def write_file(self, text):
        p = Path(self.d.name) / "Strata" / "api-key"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def test_order(self):
        self.assertIsNone(S.client_key({}))
        self.assertEqual(S.client_key({"api_key": "cfg"}), "cfg")
        self.assertEqual(S.client_key({"api_key": ["first", "second"]}), "first")
        self.write_file("from-file\r\n")
        self.assertEqual(S.client_key({"api_key": "cfg"}), "from-file")
        os.environ["STRATA_API_KEY"] = " from-env "
        self.assertEqual(S.client_key({"api_key": "cfg"}), "from-env")

    def test_an_empty_file_is_no_key(self):
        self.write_file("  \n")
        self.assertEqual(S.client_key({"api_key": "cfg"}), "cfg")


if __name__ == "__main__":
    unittest.main()
