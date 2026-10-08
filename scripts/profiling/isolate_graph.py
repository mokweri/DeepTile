#!/usr/bin/env python3
"""Extract real model operators using reference tensors bound to exported outputs."""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--nodes", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.graph.read_text())
    # Intermediate tensors are a localization aid, not an independent oracle.
    # Refuse them unless this reference run reproduces every exported output.
    for output in source["graph"]["outputs"]:
        exported = source["tensors"][source["graph"]["references"][output]]
        if digest(args.references / f"{output}.int8.bin") != digest(
                args.graph.parent / exported["file"]):
            raise SystemExit(f"Reference not bound to exported oracle: {output}")
    selected = [copy.deepcopy(node) for node in source["nodes"]
                if node["id"] in args.nodes]
    if {node["id"] for node in selected} != set(args.nodes):
        raise SystemExit("Unknown or empty node selection")
    produced = {value for node in selected for value in node["outputs"].values()}
    used = {value for node in selected for value in node["inputs"].values()}
    used.update(produced)
    external = sorted(value for value in used - produced
                      if source["tensors"][value]["kind"] != "param")
    outputs = list(selected[-1]["outputs"].values())
    graph = {key: copy.deepcopy(source[key]) for key in ("schema", "model", "target")}
    graph["model"]["name"] += "_isolated_" + "_".join(args.nodes)
    graph["graph"] = {"inputs": external, "outputs": outputs, "references": {}}
    graph["nodes"] = selected
    graph["tensors"] = {name: copy.deepcopy(source["tensors"][name]) for name in used}
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {"source_graph_sha256": digest(args.graph),
                  "nodes": args.nodes,
                  "reference_origin": "software_reference_bound_to_exported_outputs",
                  "files": {}}

    def transfer(source_path, destination):
        path = args.output / destination
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, path)
        provenance["files"][destination] = digest(path)

    for name, tensor in graph["tensors"].items():
        if name in external:
            original = tensor.get("file")
            source_path = (args.graph.parent / original if original
                           else args.references / f"{name}.int8.bin")
            tensor["kind"] = "input"
            tensor["file"] = f"inputs/{name}.int8.bin"
            transfer(source_path, tensor["file"])
        elif "file" in tensor:
            transfer(args.graph.parent / tensor["file"], tensor["file"])
    for name in outputs:
        tensor = copy.deepcopy(graph["tensors"][name])
        tensor["kind"] = "reference"
        tensor["file"] = f"refs/{name}_ref.int8.bin"
        transfer(args.references / f"{name}.int8.bin", tensor["file"])
        graph["graph"]["references"][name] = name + "_ref"
        graph["tensors"][name + "_ref"] = tensor
    (args.output / "graph.json").write_text(json.dumps(graph, indent=2) + "\n")
    (args.output / "isolation.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(args.output / "graph.json")


if __name__ == "__main__":
    main()
