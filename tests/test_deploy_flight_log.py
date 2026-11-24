"""scripts/deploy/flight_log.py: per-flight CSV + JSON sidecar, closing speed, outcome mapping (no cflib / ROS)."""
import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/deploy"))
import flight_log as fl  # noqa: E402


def st(pos, vel):
    return SimpleNamespace(pos=pos, lin_vel=vel)


class FlightLogTest(unittest.TestCase):

    def test_approach_velocity_is_closing_speed(self):
        self.assertAlmostEqual(fl.approach_velocity([0, 0, 1], [2, 0, 0], [1, 0, 1], [0, 0, 0]), 2.0)    # closing
        self.assertAlmostEqual(fl.approach_velocity([0, 0, 1], [0, 0, 0], [1, 0, 1], [1.5, 0, 0]), -1.5)  # opening
        self.assertAlmostEqual(fl.approach_velocity([0, 0, 1], [0, 3, 0], [1, 0, 1], [0, 0, 0]), 0.0)     # sideways

    def test_outcome_from_safety(self):
        cases = {"LAND: pursuer: outside the geofence at [2.4, 0, 1]": "pursuer_oob",
                 "HOLD: evader: about to leave the geofence": "evader_oob",
                 "EMERGENCY: pursuer: tilt 64 deg": "pursuer_crash",
                 "HOLD: evader: falling -2.6 m/s at 0.55 m": "evader_crash",
                 "LAND: pursuer: mocap stale inf s": "safety_stop",
                 "HOLD: non-finite policy action": "safety_stop"}
        for reason, outcome in cases.items():
            with self.subTest(reason=reason):
                self.assertEqual(fl.outcome_from_safety([reason])[0], outcome)

    def test_labels_and_file_name(self):
        self.assertEqual(fl.artifact_label("artifacts/thesis_F14_rls_s200/evader_stage23"),
                         "thesis_F14_rls_s200-evader_stage23")
        stem = fl.file_stem(0.0, "thesis_F14_rls_s200-pursuer", "a-to-b@1s")
        self.assertRegex(stem, r"^\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d-thesis_F14_rls_s200-pursuer-vs-a-to-b@1s$")
        self.assertNotIn("/", stem)
        self.assertNotIn(":", stem)

    def test_streams_rows_then_fills_outcome_and_writes_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            log = fl.FlightLog(d, "P", "E", 1000.0, {"config": {"a": 1}})
            log.row(1000.5, st([0, 0, 1], [1, 0, 0]), st([1, 0, 1], [0, 0, 0]), "E")
            self.assertTrue(os.path.isfile(log.path + ".partial"))        # crash-safe while flying
            log.row(1000.52, st([0.02, 0, 1], [1, 0, 0]), st([1, 0, 1], [0, 0, 0]), "E2")
            path = log.finish("capture", "0.09 m")
            self.assertFalse(os.path.exists(path + ".partial"))
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(list(rows[0].keys()), list(fl.COLUMNS))
            self.assertEqual([r["outcome"] for r in rows], ["capture", "capture"])
            self.assertEqual([r["evader_policy"] for r in rows], ["E", "E2"])
            self.assertAlmostEqual(float(rows[0]["t"]), 0.5)
            self.assertAlmostEqual(float(rows[0]["approach_velocity"]), 1.0)
            self.assertAlmostEqual(float(rows[0]["distance"]), 1.0)
            meta = json.load(open(path[:-4] + ".json"))
            self.assertEqual((meta["outcome"], meta["rows"], meta["config"]), ("capture", 2, {"a": 1}))

    def test_rejects_unknown_outcome(self):
        with tempfile.TemporaryDirectory() as d:
            log = fl.FlightLog(d, "P", "E", 0.0, {})
            with self.assertRaises(AssertionError):
                log.finish("win")


if __name__ == "__main__":
    unittest.main()
