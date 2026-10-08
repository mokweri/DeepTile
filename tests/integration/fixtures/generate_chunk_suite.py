#!/usr/bin/env python3
"""Combine independent wide pointwise probes into one cache-boundary graph."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from generate_epoch_probe import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    combined = None
    sources = {}
    for cigs in (128, 129, 255, 256, 257, 382, 383):
        prefix = f"cigs{cigs}"
        directory = args.output / prefix
        generate(directory, cin=cigs * 16 - 3, cout=35)
        source_path = directory / "graph.json"
        source = json.loads(source_path.read_text())
        sources[prefix] = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if combined is None:
            combined = {key: copy.deepcopy(source[key]) for key in ("schema", "model", "target")}
            combined["model"]["name"] = "partial_chain_boundaries"
            combined["graph"] = {"inputs": [], "outputs": [], "references": {}}
            combined["tensors"] = {}
            combined["nodes"] = []
        def renamed(name):
            return prefix + "_" + name
        for name, tensor in source["tensors"].items():
            if "file" in tensor:
                tensor["file"] = prefix + "/" + tensor["file"]
            combined["tensors"][renamed(name)] = tensor
        for node in source["nodes"]:
            node["id"] = renamed(node["id"])
            for field in ("inputs", "outputs"):
                node[field] = {key: renamed(value) for key, value in node[field].items()}
            combined["nodes"].append(node)
        for field in ("inputs", "outputs"):
            combined["graph"][field].extend(map(renamed, source["graph"][field]))
        combined["graph"]["references"].update({
            renamed(key): renamed(value) for key, value in source["graph"]["references"].items()})
    (args.output / "graph.json").write_text(json.dumps(combined, indent=2) + "\n")
    (args.output / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    print(args.output / "graph.json")


if __name__ == "__main__":
    main()
