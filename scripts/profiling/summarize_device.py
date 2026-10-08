#!/usr/bin/env python3
"""Validate raw AM counters against XRT and report device-active measurements."""

import argparse
import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path


CSV_FIELDS = {
    "Running Time (ms)": "execution_cycles",
    "Intra-Kernel Dataflow Stalls (ms)": "stall_internal_cycles",
    "External Memory Stalls (ms)": "stall_external_cycles",
    "Inter-Kernel Pipe Stalls (ms)": "stall_stream_cycles",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stall_row(text):
    lines = text.splitlines()
    index = lines.index("Compute Units: Stall Information")
    rows = []
    for line in lines[index + 1:]:
        if not line.strip():
            break
        rows.append(line)
    data = list(csv.DictReader(rows))
    if len(data) != 1 or data[0]["Compute Unit"] != "tile_conv_1":
        raise ValueError("Expected one DeepTile accelerator monitor row")
    return data[0]


def validate_csv(row, counters, frequency_hz):
    if int(row["Execution Count"]) != counters["execution_count"]:
        raise ValueError("XRT and raw execution counts differ")
    for field, counter in CSV_FIELDS.items():
        displayed = Decimal(row[field])
        exact_ms = Decimal(counters[counter]) * 1000 / Decimal(frequency_hz)
        # XRT uses default ostream precision (six significant digits).
        # Derive the rounding allowance from the serialized decimal place.
        tolerance = (Decimal(10) ** displayed.as_tuple().exponent) / 2
        if displayed == 0:
            tolerance = Decimal(0)
        if abs(exact_ms - displayed) > tolerance:
            raise ValueError(f"XRT/raw mismatch for {counter}: {exact_ms} vs {displayed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    context_path = args.suite / "context.json"
    context = json.loads(context_path.read_text())
    impl = json.loads(args.implementation.read_text())
    identity = context["target"]["artifact"]["sha256"]
    if impl["xclbin_sha256"] != identity or not impl["timing"]["constraints_met"]:
        raise SystemExit("Implemented-clock evidence does not match the measured target")
    frequency = impl["kernel_clock"]["frequency_hz"]
    results = []
    for record_path in sorted(args.suite.glob("*/run.json")):
        record = json.loads(record_path.read_text())
        trace = record_path.parent.with_name(record_path.parent.name + "-trace")
        raw_path = trace / "am-counters.json"
        raw = json.loads(raw_path.read_text())
        counter = raw["counters"]
        n = record["execution"]["measured_runs"] + record["execution"]["warmup_runs"]
        launches = record["execution"]["launches"]
        expected = n * launches
        if (record["status"] != "pass" or not record["results"]["exact_reference_match"] or
                record["identity"]["xclbin_sha256"] != identity or
                raw["xclbin_sha256"] != identity or raw["boot_id"] != context["boot_id"] or
                raw["reader_sha256"] != context["counter_reader_sha256"] or
                raw["boot_sha256"] != record["identity"]["boot_sha256"] or
                counter["execution_count"] != expected or counter["total_starts"] != expected or
                not raw["count_matches"] or raw["expected_execution_count"] != expected):
            raise SystemExit(f"Incomplete or mismatched counter capture: {record_path}")
        summary_path = trace / "summary.csv"
        validate_csv(stall_row(summary_path.read_text()), counter, frequency)
        result = {
            "case": record_path.parent.name, "model": record["model"],
            "launches_per_graph": launches, "graphs_including_warmup": n,
            "raw_counters": counter,
            "device_execution_cycles_per_graph": counter["execution_cycles"] / n,
            "device_busy_cycles_per_graph": counter["busy_cycles"] / n,
            "device_execution_ms_per_graph": counter["execution_cycles"] / n / frequency * 1000,
            "isolated_single_launch": launches == 1,
            "sha256": {"run": digest(record_path), "raw": digest(raw_path),
                       "xrt_summary": digest(summary_path)},
        }
        results.append(result)
        print(result["case"], "cycles/graph", result["device_execution_cycles_per_graph"],
              "busy/graph", result["device_busy_cycles_per_graph"])
    if len(results) != len(context["commands"]) or not results:
        raise SystemExit("Missing per-process counter records")
    result = {
        "schema": "deeptile.architecture-device-summary.v1",
        "context_sha256": digest(context_path),
        "implementation_sha256": digest(args.implementation),
        "xclbin_sha256": identity, "implemented_clock_hz": frequency,
        "source": "integer accelerator-monitor registers, cross-checked against XRT summary.csv",
        "scope": "all host-process launches including warmups; host timing is measured-only",
        "limitations": ["Stall counters overlap and are not a partition of device execution",
                        "Full-network aggregate is not a per-operator active-cycle attribution",
                        "Instrumented fabric; primary latency and power comparisons use uninstrumented fabric"],
        "records": results,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
