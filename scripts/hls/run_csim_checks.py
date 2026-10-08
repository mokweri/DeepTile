#!/usr/bin/env python3
"""Run an already-built HLS C model against graph oracles, preserving evidence."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("graphs", type=Path, nargs="+")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("DEEPTILE_", "TILECNN_"))}
    env.update(DEEPTILE_HLS_REPEATS=str(args.repeats), DEEPTILE_FUSE_DW="1")
    result = {"schema": "deeptile.architecture-csim-checks.v1",
              "executable": str(args.executable), "executable_sha256": digest(args.executable),
              "scope": "C model correctness only; not device timing",
              "repeats": args.repeats, "fuse_dw": True, "cases": []}
    for index, graph in enumerate(args.graphs):
        log = args.output / f"{index:02d}-{graph.parent.name}.log"
        command = [str(args.executable.resolve()), str(graph.resolve())]
        with log.open("x") as stream:
            status = subprocess.run(command, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT).returncode
        contents = log.read_text()
        output_count = len(json.loads(graph.read_text())["graph"]["outputs"])
        passed = status == 0 and contents.count("[COMPARE][EXPORTED_REF][PASS]") == output_count * args.repeats
        result["cases"].append({"graph": str(graph), "graph_sha256": digest(graph),
                                "log": log.name, "log_sha256": digest(log),
                                "command": command, "exit_status": status, "passed": passed})
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(graph.parent.name, "PASS" if passed else "FAIL", flush=True)
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
