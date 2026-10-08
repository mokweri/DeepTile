"""Regression checks for the vendor-monitor decoder (no FPGA tools needed)."""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/profiling"))

from capture_hls_evidence import decode, LOOP_FIELDS, PROCESS_FIELDS
from capture_implementation import bus_skew_summary, resources, timing_summary
from read_am_counters import sample_counters
from summarize_device import stall_row, validate_csv
from compare_runs import compare


class BoardCounterTests(unittest.TestCase):
    def test_repeat_comparison_uses_means_not_products(self):
        result = compare([9, 10, 11], [8, 8, 8])
        self.assertAlmostEqual(result["reduction_percent"], 20)
        self.assertEqual(result["baseline_over_candidate"], 1.25)
        self.assertTrue(result["all_candidate_repeat_means_faster"])

    def test_overlapping_repeats_not_claimed_separated(self):
        self.assertFalse(compare([9, 10, 11], [8, 9, 10])[
            "all_candidate_repeat_means_faster"])

    def test_latch_first_and_preserve_upper_bits(self):
        reads = []

        def read(offset):
            reads.append(offset)
            return {0x80: 8, 0x84: 17, 0xA4: 2}.get(offset, 0)

        _, registers, values = sample_counters(read)
        self.assertEqual(reads[0], 0x20)
        self.assertEqual(values["execution_count"], 8)
        self.assertEqual(values["execution_cycles"], (2 << 32) + 17)
        self.assertEqual(registers["0xa4"], 2)
        self.assertNotIn(0x08, reads)
        self.assertNotIn(0x10, reads)

    def test_reject_invalid_register_values(self):
        with self.assertRaises(ValueError):
            sample_counters(lambda _: -1)

    def test_xrt_rounding_and_count_crosscheck(self):
        row = stall_row("Compute Units: Stall Information\n"
                        "Compute Unit,Execution Count,Running Time (ms),"
                        "Intra-Kernel Dataflow Stalls (ms),External Memory Stalls (ms),"
                        "Inter-Kernel Pipe Stalls (ms),\n"
                        "tile_conv_1,8,6.67058,0.00012,0.023545,0\n\n")
        counters = dict(execution_count=8, execution_cycles=1334115,
                        stall_internal_cycles=24, stall_external_cycles=4709,
                        stall_stream_cycles=0)
        validate_csv(row, counters, 200000000)
        with self.assertRaises(ValueError):
            validate_csv(row, dict(counters, execution_count=9), 200000000)
        with self.assertRaises(ValueError):
            validate_csv(row, dict(counters, execution_cycles=1334100), 200000000)

    def test_empty_xrt_diagnostic_is_not_counter_evidence(self):
        with self.assertRaises(ValueError):
            stall_row("Debug-ip-status\nERROR: Invalid offset\n")


class MonitorDecodeTests(unittest.TestCase):
    def test_process_counter_mapping(self):
        result = decode("\n".join(map(str, range(12))) + "\n2,8;12,19\n", PROCESS_FIELDS)
        self.assertEqual(result["read_stall_cycles"], 10)
        self.assertEqual(result["write_stall_cycles"], 11)
        self.assertEqual(result["invocations"], 2)

    def test_zero_trip_invocations_are_retained(self):
        text = "0\n" * 12 + "446,446;98101,98101\n:\n"
        result = decode(text, LOOP_FIELDS, loop=True)
        self.assertEqual(result["iterations_per_invocation"], [0, 0])
        self.assertEqual(result["accepted_iterations"], 0)

    def test_loop_iteration_counts(self):
        text = "0\n" * 12 + "10,15;20,25\n10,12;11,13:20,22\n"
        result = decode(text, LOOP_FIELDS, loop=True)
        self.assertEqual(result["iterations_per_invocation"], [2, 1])
        self.assertEqual(result["accepted_iterations"], 3)

    def test_inactive_and_undefined_counters(self):
        result = decode("x\n4294967295\n" + "0\n" * 10 + "\n", LOOP_FIELDS, loop=True)
        self.assertIsNone(result["latency_avg"])
        self.assertIsNone(result["latency_max"])
        self.assertEqual(result["invocations"], 0)
        self.assertEqual(result["accepted_iterations"], 0)

    def test_incomplete_or_invalid_timestamps_are_rejected(self):
        for records in ("9,3\n", "1,4294967295\n", "2,3;4,5\n2,3\n"):
            with self.assertRaises(ValueError):
                decode("0\n" * 12 + records, LOOP_FIELDS, loop=True)


class ImplementationDecodeTests(unittest.TestCase):
    def test_routed_timing_columns(self):
        text = ("Design State      : Routed\nWNS(ns) TNS(ns) remaining columns\n"
                "------- --------\n0.101 0.0 0 261856 0.011 0.0 0 261730 0.167 0.0 0 77129\n"
                "All user specified timing constraints are met.\n")
        result = timing_summary(text)
        self.assertEqual(result["setup_endpoints"], 261856)
        self.assertEqual(result["wns_ns"], 0.101)
        self.assertEqual(result["hold_failing_endpoints"], 0)
        self.assertTrue(result["constraints_met"])
        self.assertEqual(timing_summary(text.replace("Routed", "Physopt postRoute")), result)

    def test_partial_bram_is_preserved(self):
        text = ("Design State : Routed\n"
                "| tile_conv | 12 [1%] | 3 [1%] | 15 [1%] | 4.5 [1%] | 2 [1%] | 8 [1%] |\n")
        self.assertEqual(resources(text, "tile_conv")["bram36"], 4.5)

    def test_nonrouted_reports_rejected(self):
        with self.assertRaises(ValueError):
            resources("Design State : Synthesized", "tile_conv")
        with self.assertRaises(ValueError):
            timing_summary("Design State      : Placed")

    def test_bus_skew_constraint_count_and_slack(self):
        text = ("\n1. Bus Skew Report Summary\n----------------\n"
                "1 52 [get_cells x]\n Slow 3.333 0.299 3.034\n"
                "2 54 [get_cells y]\n Slow 3.333 3.334 -0.001\n"
                "\n2. Bus Skew Report Per Constraint\n")
        result = bus_skew_summary(text)
        self.assertEqual(result["constraints"], 2)
        self.assertEqual(result["failing_constraints"], 1)
        self.assertEqual(result["minimum_slack_ns"], -0.001)
        with self.assertRaises(ValueError):
            bus_skew_summary(text.replace("Slow 3.333 3.334 -0.001", "N/A"))


if __name__ == "__main__":
    unittest.main()
