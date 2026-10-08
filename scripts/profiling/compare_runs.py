#!/usr/bin/env python3
"""Compare matched uninstrumented kernel or compiler ablations."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


CONTROLS = ("power_script_sha256", "runner_sha256", "cpu", "frequency_khz",
            "governor", "runs", "warmup", "repeats", "environment",
            "capture_profile", "xrt_ini", "timing_scope", "excluded")


def read(path):
    return json.loads(path.read_text())


def compare(before, after):
    a, b = statistics.mean(before), statistics.mean(after)
    return {"baseline_repeat_means": before, "candidate_repeat_means": after,
            "baseline_mean": a, "candidate_mean": b,
            "reduction_percent": 100 * (1 - b / a), "baseline_over_candidate": a / b,
            "all_candidate_repeat_means_faster": max(after) < min(before),
            "all_candidate_repeat_means_slower": min(after) > max(before)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--factor", choices=("kernel", "compiler"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = [read(args.baseline), read(args.candidate)]
    if summaries[0]["models"].keys() != summaries[1]["models"].keys():
        raise SystemExit("Model identities differ")
    result = {"schema": "deeptile.run-comparison.v1", "factor": args.factor,
              "models": {}, "summary_sha256": {
                  str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (args.baseline, args.candidate)},
              "note": "Host launch/wait and graph-wall measurements, not device cycles. "
                      "Repeat means are not a population confidence interval. "
                      "Implemented clock must be verified from the target separately."}
    for model in summaries[0]["models"]:
        sides = [summary["models"][model] for summary in summaries]
        contexts, raw, model_digests = [], [], []
        for side in sides:
            paths = [Path(item["record"]) for item in side]
            if not paths or len(set(paths)) != len(paths):
                raise SystemExit("Missing or duplicated run records")
            if len({p.parent.parent for p in paths}) != 1:
                raise SystemExit("A side combines different suites")
            context = read(paths[0].parent.parent / "context.json")
            records = [read(path) for path in paths]
            if context["capture_profile"] or len(records) != context["repeats"]:
                raise SystemExit("Requires complete uninstrumented suites")
            for record in records:
                if (record["status"] != "pass" or record["mode"] != "single" or
                        not record["results"].get("exact_reference_match") or
                        record["execution"]["measured_runs"] != context["runs"] or
                        record["execution"]["warmup_runs"] != context["warmup"] or
                        record["identity"]["xclbin_sha256"] != context["target"]["artifact"]["sha256"] or
                        record["identity"]["boot_sha256"] != context["target"]["boot"]["sha256"]):
                    raise SystemExit("Failed, incomplete, or unbound run record")
            contexts.append(context)
            raw.append(records)
            model_digest = records[0]["identity"].get("model_sha256")
            if not model_digest:
                fixture = read(paths[0].parent.parent / "fixture_identity.json")
                encoded = json.dumps(fixture["files"], sort_keys=True, separators=(",", ":")).encode()
                if (fixture["verification"] != "identical_before_and_after" or
                        hashlib.sha256(encoded).hexdigest() != fixture["sha256"] or
                        any(fixture["graph_path"] not in command for command in context["commands"])):
                    raise SystemExit("Unbound fixture identity")
                model_digest = fixture["sha256"]
            model_digests.append(model_digest)
        if model_digests[0] != model_digests[1]:
            raise SystemExit("Model or fixture payload differs")
        controls = CONTROLS + (("host_sha256",) if args.factor == "kernel" else ())
        for field in controls:
            if contexts[0][field] != contexts[1][field]:
                raise SystemExit(f"Unmatched control: {field}")
        for field in ("architecture", "kernel", "platform", "implementation"):
            if contexts[0]["target"][field] != contexts[1]["target"][field]:
                raise SystemExit(f"Unmatched target configuration: {field}")
        if args.factor == "compiler":
            for field in ("artifact", "boot"):
                if contexts[0]["target"][field] != contexts[1]["target"][field]:
                    raise SystemExit("Compiler ablation requires identical hardware")
        for records in raw:
            for record in records:
                reference = raw[0][0]
                if (record["identity"].get("model_sha256") != reference["identity"].get("model_sha256") or
                        record["execution"] != reference["execution"] or
                        record["results"]["compared_elements"] != reference["results"]["compared_elements"]):
                    raise SystemExit("Model, execution, or reference scope differs")
        row = {"controls": {field: contexts[0][field] for field in controls},
               "host_sha256": [c["host_sha256"] for c in contexts],
               "target_sha256": [c["target"]["artifact"]["sha256"] for c in contexts],
               "model_sha256": model_digests[0]}
        for metric in ("host_launch_sum_ms", "graph_wall_ms"):
            row[metric] = compare(*[[item[metric]["mean"] for item in side] for side in sides])
        if all("board_energy" in item for side in sides for item in side):
            row["whole_board_joules_per_inference"] = compare(*[
                [item["board_energy"]["joules_per_inference"] for item in side]
                for side in sides])
            row["minimum_sampled_power_window_fraction"] = min(
                item["board_energy"]["sampled_window_fraction"]
                for side in sides for item in side)
        result["models"][model] = row
        print(model, json.dumps({k: row[k] for k in ("host_launch_sum_ms", "graph_wall_ms")}))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
