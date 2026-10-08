#!/usr/bin/env python3
"""Generate a single-layer depthwise graph fixture for HLS validation.

Emits graph.json plus deterministic INT8 input, parameter, and reference files.
The reference arithmetic matches runtime_reference.cpp.

Usage:
  gen_dw_testcase.py <out_dir> --C 16 --H 8 --W 8 --K 3 --S 1 --pad 1 \
                     [--relu 1] [--seed 0] \
                     [--frac-in 4 --frac-w 6 --frac-b 4 --frac-out 4]
"""
import argparse
import json
import os
import numpy as np


def apply_quant(acc, bias_i32, shift_out, relu_on):
    """Integer quantization identical to reference_conv_quantize (int32 math)."""
    acc = np.int64(acc)
    if shift_out > 0:
        s1 = (acc >> (shift_out - 1)) + 1
    elif shift_out == 0:
        s1 = (acc << 1) + 1
    else:
        s1 = (acc << ((-shift_out) + 1)) + 1
    y = (s1 + np.int64(bias_i32) + 1) >> 1
    y = int(np.clip(y, -128, 127))
    if relu_on and y < 0:
        y = 0
    return y


def depthwise_ref(ifm, weight, bias_i32, C, H, W, K, S, pad,
                  shift_out, relu_on):
    OH = (H + 2 * pad - K) // S + 1
    OW = (W + 2 * pad - K) // S + 1
    out = np.zeros((C, OH, OW), dtype=np.int8)
    for c in range(C):
        for oh in range(OH):
            for ow in range(OW):
                acc = 0
                for kh in range(K):
                    for kw in range(K):
                        ih = oh * S + kh - pad
                        iw = ow * S + kw - pad
                        if 0 <= ih < H and 0 <= iw < W:
                            acc += int(ifm[c, ih, iw]) * int(weight[c, 0, kh, kw])
                out[c, oh, ow] = apply_quant(acc, int(bias_i32[c]), shift_out,
                                             relu_on)
    return out, OH, OW


def shift_bias(b_i8, shift_b):
    out = np.zeros(len(b_i8), dtype=np.int32)
    for i, v in enumerate(b_i8):
        v = np.int32(v)
        out[i] = v << shift_b if shift_b >= 0 else v >> (-shift_b)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--C", type=int, default=16)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--W", type=int, default=8)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--S", type=int, default=1)
    ap.add_argument("--pad", type=int, default=1)
    ap.add_argument("--relu", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac-in", type=int, default=4)
    ap.add_argument("--frac-w", type=int, default=6)
    ap.add_argument("--frac-b", type=int, default=4)
    ap.add_argument("--frac-out", type=int, default=4)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    C, H, W, K, S, pad = a.C, a.H, a.W, a.K, a.S, a.pad

    # Limit saturation so the fixture exercises internal datapath values.
    ifm = rng.integers(-40, 41, size=(C, H, W), dtype=np.int8)
    weight = rng.integers(-24, 25, size=(C, 1, K, K), dtype=np.int8)
    bias_i8 = rng.integers(-8, 9, size=(C,), dtype=np.int8)

    shift_out = (a.frac_in + a.frac_w) - a.frac_out
    shift_b = a.frac_out - a.frac_b + 1
    bias_i32 = shift_bias(bias_i8, shift_b)
    relu_on = bool(a.relu)

    ref, OH, OW = depthwise_ref(ifm, weight, bias_i32, C, H, W, K, S, pad,
                                shift_out, relu_on)

    inputs = os.path.join(a.out_dir, "inputs")
    params = os.path.join(a.out_dir, "params")
    refs = os.path.join(a.out_dir, "refs")
    for d in (inputs, params, refs):
        os.makedirs(d, exist_ok=True)

    ifm.tofile(os.path.join(inputs, "ifm.int8.bin"))
    weight.tofile(os.path.join(params, "dw.weight.int8.bin"))
    bias_i8.tofile(os.path.join(params, "dw.bias.int8.bin"))
    ref.tofile(os.path.join(refs, "out_ref.int8.bin"))

    graph = {
        "schema": "deeptile.graph.v1",
        "model": {"name": "dw_testcase", "producer": "gen_dw_testcase",
                  "producer_version": "0.1.0", "source_framework": "synthetic"},
        "target": {"bitwidth": 8, "signed": True, "activation_layout": "CHW",
                   "conv_weight_layout": "OIHW", "linear_weight_layout": "OI",
                   "bias_dtype": "int8"},
        "graph": {"inputs": ["ifm"], "outputs": ["dw_out"],
                  "references": {"dw_out": "dw_out_ref"}},
        "tensors": {
            "ifm": {"kind": "input", "dtype": "int8", "layout": "CHW",
                    "shape": [C, H, W], "frac": a.frac_in,
                    "file": "inputs/ifm.int8.bin"},
            "dw_out": {"kind": "activation", "dtype": "int8", "layout": "CHW",
                       "shape": [C, OH, OW], "frac": a.frac_out},
            "dw_w": {"kind": "param", "dtype": "int8", "layout": "OIHW",
                     "shape": [C, 1, K, K], "frac": a.frac_w,
                     "file": "params/dw.weight.int8.bin"},
            "dw_b": {"kind": "param", "dtype": "int8", "layout": "O",
                     "shape": [C], "frac": a.frac_b,
                     "file": "params/dw.bias.int8.bin"},
            "dw_out_ref": {"kind": "reference", "dtype": "int8", "layout": "CHW",
                           "shape": [C, OH, OW], "frac": a.frac_out,
                           "file": "refs/out_ref.int8.bin"},
        },
        "nodes": [{
            "id": "dw",
            "op": "conv2d",
            "inputs": {"ifm": "ifm", "weight": "dw_w", "bias": "dw_b"},
            "outputs": {"ofm": "dw_out"},
            "attrs": {"kernel": [K, K], "stride": [S, S],
                      "padding": [pad, pad, pad, pad], "dilation": [1, 1],
                      "groups": C},
            "post_ops": {"relu": bool(a.relu)},
        }],
    }
    with open(os.path.join(a.out_dir, "graph.json"), "w") as f:
        json.dump(graph, f, indent=2)
    print(f"[gen_dw] {a.out_dir}: C={C} {H}x{W} K{K} s{S} p{pad} "
          f"-> {OH}x{OW}, shift_out={shift_out} shift_b={shift_b}")


if __name__ == "__main__":
    main()
