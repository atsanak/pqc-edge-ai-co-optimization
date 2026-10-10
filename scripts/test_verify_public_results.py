#!/usr/bin/env python3
"""Mutation controls for the published CSV validator."""

import csv
import tempfile
import unittest
from pathlib import Path

from verify_public_results import DEFAULT_CSV, verify


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "summary.csv"
        with DEFAULT_CSV.open(newline="") as handle:
            reader = csv.DictReader(handle)
            self.fields = reader.fieldnames
            self.rows = list(reader)

    def write(self):
        with self.path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(self.rows)

    def test_published_csv(self):
        self.assertEqual(len(verify(DEFAULT_CSV)), 340)

    def test_nonfinite_metric(self):
        self.rows[0]["fps"] = "NaN"
        self.write()
        with self.assertRaisesRegex(SystemExit, "nonfinite"):
            verify(self.path)

    def test_duplicate_scheme_key(self):
        for field in ("crypto_mode", "kem_scheme", "sig_scheme"):
            self.rows[1][field] = self.rows[0][field]
        self.write()
        with self.assertRaisesRegex(SystemExit, "duplicate"):
            verify(self.path)


if __name__ == "__main__":
    unittest.main()
