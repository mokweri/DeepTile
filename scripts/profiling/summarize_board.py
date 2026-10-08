#!/usr/bin/env python3
"""Summarize measured host intervals without presenting them as device cycles."""

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def stats(values):
    ordered = sorted(values)
    return {"count": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "min": min(values),
            "max": max(values),
            "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = defaultdict(list)
    for path in sorted(args.suite.rglob("run.json")):
        record = json.loads(path.read_text())
        if record["mode"] != "single" or record["status"] != "pass":
            raise SystemExit(f"Not a passing single-image run: {path}")
        if not record["results"].get("exact_reference_match"):
            raise SystemExit(f"Exact reference comparison missing: {path}")
        launches, per_run = defaultdict(list), defaultdict(float)
        profile = path.parent / record["artifacts"]["launch_profile"]
        with profile.open() as stream:
            for row in csv.DictReader(stream):
                value = float(row["kernel_ms"])
                launches[row["launch_id"]].append(value)
                per_run[int(row["run"])] += value
        count = record["execution"]["measured_runs"]
        if len(per_run) != count or any(len(v) != count for v in launches.values()):
            raise SystemExit(f"Incomplete per-launch samples: {path}")
        if len(launches) != record["execution"]["launches"]:
            raise SystemExit(f"Launch count differs: {path}")
        result = {"record": str(path), "target": record["target"],
                  "identity": record["identity"],
                  "host_launch_sum_ms": stats(list(per_run.values())),
                  "host_launch_ms": {k: stats(v) for k, v in launches.items()}}
        if "graph_profile" in record["artifacts"]:
            graph_path = path.parent / record["artifacts"]["graph_profile"]
            with graph_path.open() as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != count:
                raise SystemExit(f"Incomplete graph-wall samples: {path}")
            for row in rows:
                stages = sum(float(row[k]) for k in
                             ("upload_ms", "execute_ms", "download_ms"))
                if abs(stages - float(row["wall_ms"])) > 0.000003:
                    raise SystemExit(f"Graph time boundaries inconsistent: {path}")
            result["graph_wall_ms"] = stats([float(row["wall_ms"]) for row in rows])
            result["graph_execute_ms"] = stats([float(row["execute_ms"]) for row in rows])
        power_path = path.parent.parent / (path.parent.name + "-power.json")
        if power_path.is_file():
            power = json.loads(power_path.read_text())
            if (power["exit_status"] != 0 or power["inferences"] != count or
                    power["run"]["record_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()):
                raise SystemExit(f"Power result not bound to this successful run: {path}")
            result["board_energy"] = {
                "record": str(power_path),
                "joules_per_inference": power["joules_per_inference"],
                "dynamic_joules_per_inference": power.get("dynamic_joules_per_inference"),
                "mean_w": power["load"]["mean_w"],
                "sample_rate_hz": power["load"]["effective_rate_hz"],
                "sensor": power["sensor"],
                "baseline_drift": power["baseline_drift"],
                "sampled_window_fraction": power["load"]["duration_s"] / power["window"]["duration_s"],
                "boundary_note": "Existing sampler integrates first-to-last in-window samples, without endpoint extrapolation",
            }
        groups[record["model"]].append(result)
    if not groups:
        raise SystemExit("No run records found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "deeptile.architecture-host-summary.v1",
                                      "models": groups}, indent=2) + "\n")
    for model, records in groups.items():
        print(model)
        for record in records:
            print(" ", record["target"],
                  "launch_sum", round(record["host_launch_sum_ms"]["mean"], 6),
                  "graph_wall", record.get("graph_wall_ms", {}).get("mean", "unmeasured"))
            if "board_energy" in record:
                print("   board_J/inference", record["board_energy"]["joules_per_inference"],
                      "sampled_window_fraction", record["board_energy"]["sampled_window_fraction"])


if __name__ == "__main__":
    main()
