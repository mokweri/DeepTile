#!/usr/bin/env python3
"""Generate independent signed-integer wide/odd-stride loader regression oracles.

Four branches exercise 199/200-wide inputs, with and without fused 3x3 pooling,
17 input channels, 19 output channels, spatial tiling, and channel tails.
The scalar convolution uses the tensor quantization scales directly; it does
not invoke the DeepTile compiler, packed-layout routines, or software reference.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path


def generate(output):
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(200199)
    cin, cout, height = 17, 19, 5
    weights = [rng.randrange(-12, 13) for _ in range(cout * cin * 9)]
    biases = [rng.randrange(-64, 65) for _ in range(cout)]
    tensors, nodes, inputs, outputs, refs, hashes = {}, [], [], [], {}, {}

    def tensor(name, kind, layout, shape, frac, values=None):
        record = {"kind": kind, "dtype": "int8", "layout": layout,
                  "shape": shape, "frac": frac}
        if values is not None:
            data = bytes(value & 255 for value in values)
            record["file"] = name + ".int8.bin"
            (output / record["file"]).write_bytes(data)
            hashes[record["file"]] = hashlib.sha256(data).hexdigest()
        tensors[name] = record

    tensor("weights", "param", "OIHW", [cout, cin, 3, 3], 6, weights)
    tensor("bias", "param", "O", [cout], 5, biases)
    for width in (199, 200):
        values = [rng.randrange(-32, 33) for _ in range(cin * height * width)]
        name = f"input_w{width}"
        inputs.append(name)
        tensor(name, "input", "CHW", [cin, height, width], 4, values)
        conv = []
        for oc in range(cout):
            for y in range(height):
                for x in range(width):
                    accumulator = biases[oc] * 32  # frac 5 -> product frac 10
                    for ic in range(cin):
                        for ky in range(3):
                            iy = y + ky - 1
                            if not 0 <= iy < height:
                                continue
                            for kx in range(3):
                                ix = x + kx - 1
                                if 0 <= ix < width:
                                    accumulator += (values[(ic * height + iy) * width + ix]
                                                    * weights[((oc * cin + ic) * 3 + ky) * 3 + kx])
                    # Product frac 10 -> output frac 4; half-up signed rounding.
                    conv.append(max(-128, min(127, (accumulator + 32) // 64)))
        for pooled in (False, True):
            branch = f"w{width}_{'pooled' if pooled else 'unpooled'}"
            conv_id = branch + "_conv"
            conv_out = conv_id + "_out"
            tensor(conv_out, "activation", "CHW", [cout, height, width], 4)
            nodes.append({"id": conv_id, "op": "conv2d",
                          "inputs": {"ifm": name, "weight": "weights", "bias": "bias"},
                          "outputs": {"ofm": conv_out},
                          "attrs": {"kernel": [3, 3], "stride": [1, 1],
                                    "padding": [1, 1, 1, 1], "dilation": [1, 1], "groups": 1},
                          "post_ops": {"relu": pooled}})
            final, reference, oh, ow = conv_out, conv, height, width
            if pooled:
                oh, ow = (height + 1) // 2, (width + 1) // 2
                reference = []
                for oc in range(cout):
                    for y in range(oh):
                        for x in range(ow):
                            neighborhood = [conv[(oc * height + iy) * width + ix]
                                            for iy in range(2 * y - 1, 2 * y + 2)
                                            for ix in range(2 * x - 1, 2 * x + 2)
                                            if 0 <= iy < height and 0 <= ix < width]
                            reference.append(max(0, max(neighborhood)))
                final = branch + "_out"
                tensor(final, "activation", "CHW", [cout, oh, ow], 4)
                nodes.append({"id": branch + "_pool", "op": "maxpool2d",
                              "inputs": {"ifm": conv_out}, "outputs": {"ofm": final},
                              "attrs": {"kernel": [3, 3], "stride": [2, 2],
                                        "padding": [1, 1, 1, 1]}})
            outputs.append(final)
            refs[final] = final + "_ref"
            tensor(refs[final], "reference", "CHW", [cout, oh, ow], 4, reference)
    graph = {"schema": "deeptile.graph.v1",
             "model": {"name": "raw_loader_boundaries", "producer": "scalar_integer_probe",
                       "producer_version": "1.0.0", "source_framework": "independent_python"},
             "target": {"bitwidth": 8, "signed": True, "activation_layout": "CHW",
                        "conv_weight_layout": "OIHW", "linear_weight_layout": "OI", "bias_dtype": "int8"},
             "graph": {"inputs": inputs, "outputs": outputs, "references": refs},
             "tensors": tensors, "nodes": nodes}
    (output / "graph.json").write_text(json.dumps(graph, indent=2) + "\n")
    hashes["graph.json"] = hashlib.sha256((output / "graph.json").read_bytes()).hexdigest()
    (output / "provenance.json").write_text(json.dumps({
        "seed": 200199, "oracle": "independent scalar convolution, scale conversion and pooling",
        "files": hashes}, indent=2) + "\n")
    print(output / "graph.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    generate(parser.parse_args().output)
