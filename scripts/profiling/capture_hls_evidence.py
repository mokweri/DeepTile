#!/usr/bin/env python3
"""Preserve RTL co-simulation evidence and summarize Vitis 2024.2 monitors.

Monitor counters overlap and are not a partition of kernel latency. Read stalls
include descriptor/input starvation, not specifically AXI memory service time.
The monitor's loop iteration count is accepted iterations, not scalar MACs.
"""

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import zipfile
from pathlib import Path


PROCESS_FIELDS = (
    "total_stall_cycles", "observation_cycles", "invocation_ii_avg",
    "invocation_ii_max", "invocation_ii_min", "latency_avg", "latency_max",
    "latency_min", "no_start_stall_cycles", "no_continue_stall_cycles",
    "read_stall_cycles", "write_stall_cycles",
)
LOOP_FIELDS = (
    "latency_avg", "latency_max", "latency_min", "invocation_ii_avg",
    "invocation_ii_max", "invocation_ii_min", "iteration_latency_avg",
    "iteration_latency_max", "iteration_latency_min", "iteration_ii_avg",
    "iteration_ii_max", "iteration_ii_min",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pairs(line):
    if not line.strip():
        return []
    result = [tuple(map(int, entry.split(","))) for entry in line.split(";")]
    if any(len(entry) != 2 or entry[1] < entry[0] or 4294967295 in entry
           for entry in result):
        raise ValueError("Invalid/unfinished monitor timestamps")
    return result


def decode(text, fields, loop=False):
    lines = text.splitlines()
    # Some inactive/single-iteration loops leave an II statistic undefined.
    # Preserve it as unknown, never as a zero-latency observation.
    raw = [None if value.strip().lower() in {"x", "z"} else int(value)
           for value in lines[:12]]
    if len(raw) != 12:
        raise ValueError("Expected 12 monitor counters")
    windows = pairs(lines[12]) if len(lines) > 12 else []
    result = {name: (None if value == 4294967295 else value)
              for name, value in zip(fields, raw)}
    result["invocations"] = len(windows)
    result["invocation_windows"] = windows
    if loop:
        # Empty colon-separated groups are real zero-trip invocations (e.g.
        # the ping-pong steady loop when num_cigs=1), not missing records.
        iterations = ([pairs(group) for group in lines[13].split(":")]
                      if windows and len(lines) > 13 else [])
        result["iterations_per_invocation"] = [len(group) for group in iterations]
        result["accepted_iterations"] = sum(map(len, iterations))
        if windows and len(iterations) != len(windows):
            raise ValueError("Loop invocation/iteration groups differ")
    return result


def summarize(project):
    solution = project / "hls"
    db = solution / ".autopilot/db"
    report = (solution / "sim/report/tile_conv_cosim.rpt").read_text()
    if not re.search(r"\|\s*Verilog\s*\|\s*Pass\s*\|", report):
        raise ValueError("No passing Verilog co-simulation report")
    transaction_text = (solution / "sim/verilog/tile_conv.performance.result.transaction.xml").read_text()
    transactions = [{"index": int(index), "active_cycles": int(latency),
                     "interval_cycles": None if interval == "x" else int(interval)}
                    for index, latency, interval in re.findall(
                        r"transaction\s+(\d+):\s+(\d+)\s+(\d+|x)", transaction_text)]
    if not transactions:
        raise ValueError("Missing device-active RTL transactions")
    processes = {}
    with zipfile.ZipFile(db / "process_stalling_info/process.zip") as archive:
        for parent, name, _, status in csv.reader(io.StringIO((db / "process_info.csv").read_text())):
            if not status.startswith("status"):
                continue
            processes[f"{parent}.{name}".lstrip(".")] = decode(
                archive.read(status).decode(), PROCESS_FIELDS)
    loops = {}
    with zipfile.ZipFile(db / "loop_performance_info/loop.zip") as archive:
        for parent, name, status in csv.reader(io.StringIO((db / "loop_info.csv").read_text())):
            loops[f"{parent}.{name}"] = decode(
                archive.read(status).decode(), LOOP_FIELDS, loop=True)
    return {"schema": "deeptile.architecture-rtl-summary.v1",
            "measurement": "RTL co-simulation; not board or host timing",
            "counter_scope": "Overlapping process stall observations; no inferred cycle partition",
            "transactions": transactions, "processes": processes, "loops": loops}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-note", required=True,
                        help="Source/configuration provenance; do not infer it from current files")
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.project)
    summary["source_note"] = args.source_note
    summary["original_project"] = str(args.project)
    solution = args.project / "hls"
    files = ["sim/report/tile_conv_cosim.rpt",
             "sim/verilog/tile_conv.performance.result.transaction.xml",
             "syn/report/tile_conv_csynth.rpt", "syn/report/tile_conv_csynth.xml",
             ".autopilot/db/process_info.csv", ".autopilot/db/loop_info.csv",
             ".autopilot/db/channel_info.csv",
             ".autopilot/db/process_stalling_info/process.zip",
             ".autopilot/db/loop_performance_info/loop.zip",
             ".autopilot/db/channel_depth_info/channel.zip"]
    # Retain tool implementation hashes for auditable counter field decoding.
    monitors = ["df_process_monitor.svh", "loop_sample_agent.svh", "csv_file_dump.svh",
                "dataflow_monitor.sv", "loop_performance_monitor.sv"]
    summary["monitor_sha256"] = {name: digest(solution / "sim/verilog" / name)
                                 for name in monitors
                                 if (solution / "sim/verilog" / name).is_file()}
    for source in [args.log, *(solution / name for name in files)]:
        if not source.is_file():
            raise FileNotFoundError(source)
    args.output.mkdir(parents=True, exist_ok=False)
    preserved = {}
    for relative in files:
        source = solution / relative
        target = args.output / source.name
        shutil.copy2(source, target)
        preserved[target.name] = digest(target)
    shutil.copy2(args.log, args.output / "hls.log")
    preserved["hls.log"] = digest(args.output / "hls.log")
    summary["preserved_sha256"] = preserved
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("RTL active cycles:", [row["active_cycles"] for row in summary["transactions"]])
    for name, row in summary["processes"].items():
        print(name, "read_stall", row["read_stall_cycles"],
              "write_stall", row["write_stall_cycles"])
    for name, row in summary["loops"].items():
        if any(part in name for part in ("RAW_", "MAC_FLAT", "EMIT_LOOP")):
            print(name, "loop_latency", row["latency_avg"],
                  "iterations", row["accepted_iterations"])


if __name__ == "__main__":
    main()
