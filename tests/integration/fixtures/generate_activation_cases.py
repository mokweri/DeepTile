#!/usr/bin/env python3
"""Generate independent signed-integer oracles for clamped activations and 5x5 kernels.

Seven cases cover the ReLU6 clamp on a dense convolution, on fused
pointwise-to-depthwise pairs with 3x3 and 5x5 kernels, and after a residual
add, plus 5x5 dense and depthwise convolutions, one of them wide enough to span
several column tiles.
Every case is constructed so that its clamp is reached. The scalar oracle
applies the tensor quantization scales directly; it does not invoke the
DeepTile compiler, packed-layout routines, or software reference.
"""

import argparse
import json
import random
from pathlib import Path

TARGET = {"bitwidth": 8, "signed": True, "activation_layout": "CHW",
          "conv_weight_layout": "OIHW", "linear_weight_layout": "OI",
          "bias_dtype": "int8"}


def relu6_max(frac):
    return min(127, round(6.0 * 2 ** frac))


def saturate(value):
    return max(-128, min(127, value))


def activate(value, post, frac, after_add):
    relu6 = post.get("post_add_relu6" if after_add else "relu6", False)
    relu = post.get("post_add_relu" if after_add else "relu", False)
    if relu6:
        return max(0, min(relu6_max(frac), value))
    return max(0, value) if relu else value


def conv(ifm, shape, weights, bias, cout, k, stride, groups, fracs, post):
    """Scalar convolution with the accelerator's two-stage requantization."""
    cin, height, width = shape
    fin, fw, fb, fout = fracs
    pad = k // 2
    oh = (height + 2 * pad - k) // stride + 1
    ow = (width + 2 * pad - k) // stride + 1
    shift = fin + fw - fout
    bias_shift = fout - fb + 1
    out = []
    for oc in range(cout):
        channels = [oc] if groups > 1 else range(cin)
        aligned = bias[oc] << bias_shift if bias_shift >= 0 else bias[oc] >> -bias_shift
        for oy in range(oh):
            for ox in range(ow):
                acc = 0
                for slot, ic in enumerate(channels):
                    for ky in range(k):
                        iy = oy * stride + ky - pad
                        if not 0 <= iy < height:
                            continue
                        for kx in range(k):
                            ix = ox * stride + kx - pad
                            if 0 <= ix < width:
                                wi = ((oc * len(channels) + slot) * k + ky) * k + kx
                                acc += ifm[(ic * height + iy) * width + ix] * weights[wi]
                if shift > 0:
                    s1 = acc >> (shift - 1)
                elif shift == 0:
                    s1 = acc << 1
                else:
                    s1 = acc << (1 - shift)
                value = saturate((s1 + aligned + 1) >> 1)
                if not post.get("residual_add", False):
                    value = activate(value, post, fout, after_add=False)
                out.append(value)
    return out, [cout, oh, ow]


def signed_shift(value, shift):
    if shift > 0:
        return value << shift
    if shift < 0:
        s = -shift
        return (value + (1 << (s - 1)) - (1 if value < 0 else 0)) >> s
    return value


class Case:
    def __init__(self, root, name, shape, seed):
        self.root, self.name = root / name, name
        self.rng = random.Random(seed)
        self.tensors, self.nodes, self.values = {}, [], {}
        self.current = self.tensor("input", "input", "CHW", shape, 5,
                                   [self.rng.randrange(-128, 128) for _ in range(_count(shape))])

    def tensor(self, name, kind, layout, shape, frac, values=None, reference=False):
        record = {"kind": kind, "dtype": "int8", "layout": layout,
                  "shape": list(shape), "frac": frac}
        if values is not None:
            record["file"] = name + ".int8.bin"
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / record["file"]).write_bytes(bytes(v & 255 for v in values))
            if not reference:
                self.values[name] = values
        self.tensors[name] = record
        return name

    def conv(self, name, cout, k, stride, out_frac, post, groups=1, residual=None, wmax=40):
        shape = self.tensors[self.current]["shape"]
        wshape = [cout, 1 if groups > 1 else shape[0], k, k]
        weights = [self.rng.randrange(-wmax, wmax + 1) for _ in range(_count(wshape))]
        bias = [self.rng.randrange(-60, 61) for _ in range(cout)]
        self.tensor(name + "_w", "param", "OIHW", wshape, 7, weights)
        self.tensor(name + "_b", "param", "O", [cout], 5, bias)
        fracs = (self.tensors[self.current]["frac"], 7, 5, out_frac)
        values, oshape = conv(self.values[self.current], shape, weights, bias, cout, k,
                              stride, groups, fracs, post)
        inputs = {"ifm": self.current, "weight": name + "_w", "bias": name + "_b"}
        if residual:
            inputs["residual"] = residual
            rshift = out_frac - self.tensors[residual]["frac"]
            values = [activate(saturate(v + signed_shift(r, rshift)), post, out_frac, True)
                      for v, r in zip(values, self.values[residual])]
        out = name + "_out"
        self.tensors[out] = {"kind": "activation", "dtype": "int8", "layout": "CHW",
                             "shape": oshape, "frac": out_frac}
        self.values[out] = values
        self.nodes.append({"id": name, "op": "conv2d", "inputs": inputs,
                           "outputs": {"ofm": out},
                           "attrs": {"kernel": [k, k], "stride": [stride, stride],
                                     "padding": [k // 2] * 4, "dilation": [1, 1],
                                     "groups": groups},
                           "post_ops": post})
        self.current = out
        return out

    def write(self):
        out = self.current
        self.tensor(out + "_ref", "reference", "CHW", self.tensors[out]["shape"],
                    self.tensors[out]["frac"], self.values[out], reference=True)
        graph = {"schema": "deeptile.graph.v1", "model": {"name": self.name},
                 "target": TARGET,
                 "graph": {"inputs": ["input"], "outputs": [out],
                           "references": {out: out + "_ref"}},
                 "tensors": self.tensors, "nodes": self.nodes}
        (self.root / "graph.json").write_text(json.dumps(graph, indent=1) + "\n")


def _count(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total


def generate(output):
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    case = Case(output, "relu6_dense_k3", [16, 16, 16], 1)        # clamp 96
    case.conv("conv", 32, 3, 1, 4, {"relu6": True})
    cases.append(case)
    case = Case(output, "relu6_fused_pwdw", [16, 16, 16], 2)      # clamps 96, 48
    case.conv("pw", 32, 1, 1, 4, {"relu6": True})
    case.conv("dw", 32, 3, 1, 3, {"relu6": True}, groups=32, wmax=127)
    cases.append(case)
    case = Case(output, "relu6_fused_pwdw5", [16, 36, 20], 7)     # clamps 96, 48
    case.conv("pw", 48, 1, 1, 4, {"relu6": True})                 # two row tiles
    case.conv("dw", 48, 5, 1, 3, {"relu6": True}, groups=48, wmax=127)
    cases.append(case)
    case = Case(output, "relu6_post_add", [32, 12, 12], 3)        # clamp 96
    case.conv("proj", 32, 1, 1, 4, {"residual_add": True, "post_add_relu6": True},
              residual=case.current)
    cases.append(case)
    case = Case(output, "k5_dense", [16, 20, 20], 4)
    case.conv("conv", 16, 5, 1, 4, {"relu": True})
    cases.append(case)
    case = Case(output, "k5_depthwise_s2", [32, 28, 28], 5)
    case.conv("dw", 32, 5, 2, 4, {"relu6": True}, groups=32)
    cases.append(case)
    case = Case(output, "k5_depthwise_wide", [16, 20, 150], 6)    # several column tiles
    case.conv("dw", 16, 5, 1, 4, {"relu6": True}, groups=16)
    cases.append(case)
    for case in cases:
        case.write()
    return [case.root / "graph.json" for case in cases]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    for graph in generate(parser.parse_args().output):
        print(graph)


if __name__ == "__main__":
    main()
