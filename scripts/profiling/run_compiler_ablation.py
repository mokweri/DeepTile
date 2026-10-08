#!/usr/bin/env python3
"""Run a matched compiler ablation and bind unversioned graph payloads to it."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def snapshot(graph):
    tensors = json.loads(graph.read_text())["tensors"]
    files = {"graph.json": hashlib.sha256(graph.read_bytes()).hexdigest()}
    for tensor in tensors.values():
        if "file" in tensor:
            path = (graph.parent / tensor["file"]).resolve()
            path.relative_to(graph.parent)
            files[tensor["file"]] = hashlib.sha256(path.read_bytes()).hexdigest()
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"graph_path": str(graph), "files": files,
            "sha256": hashlib.sha256(encoded).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--baseline-host", type=Path, required=True)
    parser.add_argument("--candidate-host", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    graph = args.graph.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    before = snapshot(graph)
    (args.output / "inputs-before.json").write_text(json.dumps(before, indent=2) + "\n")
    for side, host in (("baseline", args.baseline_host), ("candidate", args.candidate_host)):
        command = [sys.executable, str(args.runner.resolve()), "--host", str(host.resolve()),
                   "--graphs", str(graph), "--runs", str(args.runs),
                   "--warmup", str(args.warmup), "--repeats", str(args.repeats),
                   "--output", str((args.output / side).resolve())]
        subprocess.run(command, check=True)
        after = snapshot(graph)
        (args.output / f"inputs-after-{side}.json").write_text(json.dumps(after, indent=2) + "\n")
        if before != after:
            raise SystemExit("Fixture changed during ablation; comparison is invalid")
        identity = dict(before, verification="identical_before_and_after",
                        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (args.output / side / "fixture_identity.json").write_text(json.dumps(identity, indent=2) + "\n")


if __name__ == "__main__":
    main()
