"""Small gates for ablation arithmetic and fixture identity binding."""

import json
import tempfile
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/profiling"))

from compare_runs import compare
from run_compiler_ablation import snapshot


class MeasurementTests(unittest.TestCase):
    def test_faster_and_slower_are_distinct(self):
        faster = compare([10, 10, 10], [2, 2, 2])
        self.assertEqual(faster["reduction_percent"], 80)
        self.assertEqual(faster["baseline_over_candidate"], 5)
        self.assertTrue(faster["all_candidate_repeat_means_faster"])
        self.assertFalse(faster["all_candidate_repeat_means_slower"])
        slower = compare([2, 2, 2], [10, 10, 10])
        self.assertEqual(slower["reduction_percent"], -400)
        self.assertFalse(slower["all_candidate_repeat_means_faster"])
        self.assertTrue(slower["all_candidate_repeat_means_slower"])

    def test_overlapping_repeats(self):
        result = compare([9, 10, 11], [10, 10, 10])
        self.assertEqual(result["reduction_percent"], 0)
        self.assertFalse(result["all_candidate_repeat_means_faster"])
        self.assertFalse(result["all_candidate_repeat_means_slower"])

    def test_snapshot_binds_graph_and_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            graph = root / "graph.json"
            graph.write_text(json.dumps({"tensors": {"x": {"file": "x.bin"}}}))
            payload = root / "x.bin"
            payload.write_bytes(b"\x01")
            before = snapshot(graph)
            self.assertEqual(before, snapshot(graph))
            payload.write_bytes(b"\x02")
            self.assertNotEqual(before["sha256"], snapshot(graph)["sha256"])


if __name__ == "__main__":
    unittest.main()
