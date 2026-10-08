#!/usr/bin/env python3
"""Generate a small independent integer oracle for bounded descriptor-FIFO tests.

Use DEEPTILE_FORCE_EPOCHS to vary the transfer count without changing tensors.
This synthetic schedule tests progress, not representative model performance.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path


def generate(output, cin=16, cout=4096, height=1, width=1):
    if min(cin, cout, height, width) < 1:
        raise ValueError("channel counts and spatial dimensions must be positive")
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(25632)
    spatial = height * width
    inputs = [rng.randrange(-16, 17) for _ in range(cin * spatial)]
    weights = [rng.randrange(-8, 9) for _ in range(cout * cin)]
    biases = [rng.randrange(-32, 33) for _ in range(cout)]
    # All tensors have frac=4. Bias is aligned to product frac=8, then
    # rounded half-up and saturated back to signed INT8 at frac=4.
    reference = [max(-128, min(127, (16 * biases[oc] + sum(
        inputs[ic * spatial + pixel] * weights[oc * cin + ic]
        for ic in range(cin)) + 8) // 16))
        for oc in range(cout) for pixel in range(spatial)]
    tensors, hashes = {}, {}

    def tensor(name, kind, layout, shape, values=None):
        item = {"kind": kind, "dtype": "int8", "layout": layout,
                "shape": shape, "frac": 4}
        if values is not None:
            data = bytes(value & 255 for value in values)
            item["file"] = name + ".int8.bin"
            (output / item["file"]).write_bytes(data)
            hashes[item["file"]] = hashlib.sha256(data).hexdigest()
        tensors[name] = item

    tensor("input", "input", "CHW", [cin, height, width], inputs)
    tensor("weights", "param", "OIHW", [cout, cin, 1, 1], weights)
    tensor("bias", "param", "O", [cout], biases)
    tensor("output", "activation", "CHW", [cout, height, width])
    tensor("output_ref", "reference", "CHW", [cout, height, width], reference)
    graph = {
        "schema": "deeptile.graph.v1",
        "model": {"name": "descriptor_epoch_probe", "producer": "scalar_integer_probe",
                  "producer_version": "1.0.0", "source_framework": "independent_python"},
        "target": {"bitwidth": 8, "signed": True, "activation_layout": "CHW",
                   "conv_weight_layout": "OIHW", "linear_weight_layout": "OI",
                   "bias_dtype": "int8"},
        "graph": {"inputs": ["input"], "outputs": ["output"],
                  "references": {"output": "output_ref"}},
        "tensors": tensors,
        "nodes": [{"id": "epoch_probe", "op": "conv2d",
                   "inputs": {"ifm": "input", "weight": "weights", "bias": "bias"},
                   "outputs": {"ofm": "output"},
                   "attrs": {"kernel": [1, 1], "stride": [1, 1],
                             "padding": [0, 0, 0, 0], "dilation": [1, 1], "groups": 1},
                   "post_ops": {"relu": False}}]}
    graph_bytes = (json.dumps(graph, indent=2) + "\n").encode()
    (output / "graph.json").write_bytes(graph_bytes)
    hashes["graph.json"] = hashlib.sha256(graph_bytes).hexdigest()
    (output / "provenance.json").write_text(json.dumps({
        "seed": 25632, "oracle": "independent scalar signed integer dot product",
        "max_forced_epochs": (cout + 15) // 16, "files": hashes}, indent=2) + "\n")
    print(output / "graph.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cin", type=int, default=16)
    parser.add_argument("--cout", type=int, default=4096)
    parser.add_argument("--height", type=int, default=1)
    parser.add_argument("--width", type=int, default=1)
    args = parser.parse_args()
    if min(args.cin, args.cout, args.height, args.width) < 1:
        parser.error("channel counts and spatial dimensions must be positive")
    generate(args.output, args.cin, args.cout, args.height, args.width)
