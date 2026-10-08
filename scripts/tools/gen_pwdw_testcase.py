#!/usr/bin/env python3
"""Generate fused pointwise-depthwise graph fixtures for HLS validation.

Each deterministic fixture contains a multi-node graph, INT8 parameters, and a
bit-exact final reference. Cases cover stride, ragged channels, row tiling,
non-pointwise producers, and a MobileNetV2 residual block, with 3x3 and 5x5
depthwise kernels.

The seed is derived from the case name with CRC-32. Fixtures generated before
that change used Python's per-process randomized ``hash()`` and cannot be
regenerated bit for bit; keep their checked-in files.

Usage:
  gen_pwdw_testcase.py <out_root> [--case all|single|...] [--seed 0]
"""
import argparse
import json
import math
import os
import zlib
import numpy as np

# Per-tensor quantization shared by every layer in a fixture.
FRAC_ACT = 4        # every activation tensor (input, intermediates, output)
FRAC_W = 6          # every weight tensor
FRAC_B = 4          # every bias tensor
SHIFT_OUT = (FRAC_ACT + FRAC_W) - FRAC_ACT   # = FRAC_W = 6, for every conv launch
SHIFT_B = FRAC_ACT - FRAC_B + 1              # = 1
RES_SHIFT = FRAC_ACT - FRAC_ACT              # = 0 (residual same scale as output)

IFM_RANGE = 40      # input activation magnitude
BIAS_RANGE = 8      # int8 bias magnitude


def apply_quant(acc, bias_i32, relu_on):
    """int32 requant identical to reference_conv_quantize (SHIFT_OUT>0 branch)."""
    acc = np.int64(acc)
    # INVARIANT: The +1 below is the only round-half-up bias; adding a bias before
    # this shift changes the bit-exact result.
    s1 = acc >> (SHIFT_OUT - 1)
    y = (s1 + np.int64(bias_i32) + 1) >> 1
    y = int(np.clip(y, -128, 127))          # explicit clamp
    if relu_on and y < 0:                   # explicit-clamp ReLU
        y = 0
    return y


def apply_signed_shift(val, shift):
    """Identical to runtime_reference.cpp::apply_signed_shift (round-half-up)."""
    val = np.int32(val)
    if shift > 0:
        return int(val << shift)
    if shift < 0:
        s = -shift
        round_bias = (np.int32(1) << (s - 1)) + (val >> 31)
        return int((val + round_bias) >> s)
    return int(val)


def shift_bias(b_i8):
    return np.array([int(np.int32(v) << SHIFT_B) if SHIFT_B >= 0
                     else int(np.int32(v) >> (-SHIFT_B)) for v in b_i8],
                    dtype=np.int32)


def weight_range(reduction):
    """Reduction-scaled weight magnitude so the accumulator RMS is ~constant
    regardless of the reduction depth (matches gen_dw's ±24 at reduction=9)."""
    return max(3, int(round(72.0 / math.sqrt(reduction))))


# INT8 reference convolution for dense and depthwise layers.
def conv_ref(ifm, weight, bias_i32, Cin, H, W, Cout, K, S, pad, depthwise,
             relu_on):
    OH = (H + 2 * pad - K) // S + 1
    OW = (W + 2 * pad - K) // S + 1
    out = np.zeros((Cout, OH, OW), dtype=np.int8)
    for oc in range(Cout):
        ic_lo, ic_hi = (oc, oc + 1) if depthwise else (0, Cin)
        for oh in range(OH):
            for ow in range(OW):
                acc = 0
                for ic in range(ic_lo, ic_hi):
                    for kh in range(K):
                        for kw in range(K):
                            ih = oh * S + kh - pad
                            iw = ow * S + kw - pad
                            if 0 <= ih < H and 0 <= iw < W:
                                if depthwise:
                                    wv = int(weight[oc, 0, kh, kw])
                                else:
                                    wv = int(weight[oc, ic, kh, kw])
                                acc += int(ifm[ic, ih, iw]) * wv
                out[oc, oh, ow] = apply_quant(acc, int(bias_i32[oc]), relu_on)
    return out, OH, OW


def residual_ref(main_i8, res_i8, post_add_relu):
    out = np.zeros_like(main_i8)
    flat_m = main_i8.reshape(-1)
    flat_r = res_i8.reshape(-1)
    flat_o = out.reshape(-1)
    for i in range(flat_m.size):
        s = int(flat_m[i]) + apply_signed_shift(int(flat_r[i]), RES_SHIFT)
        s = int(np.clip(s, -128, 127))
        if post_add_relu and s < 0:
            s = 0
        flat_o[i] = s
    return out


# Layer dictionaries define the graph chain and post-operations.
def build_layers(case):
    if case == "single":
        return (32, 16, 16), [
            dict(id="pw", kind="conv", Cout=32, K=1, S=1, pad=0, relu=True),
            dict(id="dw", kind="dw",   Cout=32, K=3, S=1, pad=1, relu=True),
        ]
    if case == "stride2":
        return (32, 15, 15), [
            dict(id="pw", kind="conv", Cout=32, K=1, S=1, pad=0, relu=True),
            dict(id="dw", kind="dw",   Cout=32, K=3, S=2, pad=1, relu=True),
        ]
    if case == "ragged":
        return (24, 16, 16), [
            dict(id="pw", kind="conv", Cout=24, K=1, S=1, pad=0, relu=True),
            dict(id="dw", kind="dw",   Cout=24, K=3, S=1, pad=1, relu=True),
        ]
    if case == "rowtiled":
        return (32, 56, 56), [
            dict(id="pw", kind="conv", Cout=32, K=1, S=1, pad=0, relu=True),
            dict(id="dw", kind="dw",   Cout=32, K=3, S=1, pad=1, relu=True),
        ]
    if case == "after_k3":
        return (3, 16, 16), [
            dict(id="conv", kind="conv", Cout=32, K=3, S=1, pad=1, relu=True),
            dict(id="dw",   kind="dw",   Cout=32, K=3, S=1, pad=1, relu=True),
        ]
    if case == "v2_block":
        return (24, 16, 16), [
            dict(id="expand",  kind="conv", Cout=144, K=1, S=1, pad=0, relu=True),
            dict(id="dw",      kind="dw",   Cout=144, K=3, S=1, pad=1, relu=True),
            # MobileNetV2 projects and adds the residual without ReLU.
            dict(id="project", kind="conv", Cout=24,  K=1, S=1, pad=0, relu=False,
                 residual=True, post_add_relu=False),
        ]
    # 5x5 depthwise: same padding 2, EfficientNet-Lite's kernel.
    if case == "k5_single":
        return (32, 16, 16), [
            dict(id="pw", kind="conv", Cout=32, K=1, S=1, pad=0, relu=True),
            dict(id="dw", kind="dw",   Cout=32, K=5, S=1, pad=2, relu=True),
        ]
    if case == "k5_stride2":
        return (32, 15, 15), [
            dict(id="pw", kind="conv", Cout=32, K=1, S=1, pad=0, relu=True),
            dict(id="dw", kind="dw",   Cout=32, K=5, S=2, pad=2, relu=True),
        ]
    if case == "k5_ragged":
        return (24, 16, 16), [
            dict(id="pw", kind="conv", Cout=40, K=1, S=1, pad=0, relu=True),
            dict(id="dw", kind="dw",   Cout=40, K=5, S=1, pad=2, relu=True),
        ]
    if case == "k5_rowtiled":             # two row tiles: 4-row halo seam
        return (32, 56, 56), [
            dict(id="pw", kind="conv", Cout=32, K=1, S=1, pad=0, relu=True),
            dict(id="dw", kind="dw",   Cout=32, K=5, S=1, pad=2, relu=True),
        ]
    if case == "k5_rowtiled_s2":          # three row tiles at stride 2
        return (16, 60, 60), [
            dict(id="pw", kind="conv", Cout=32, K=1, S=1, pad=0, relu=True),
            dict(id="dw", kind="dw",   Cout=32, K=5, S=2, pad=2, relu=True),
        ]
    if case == "k5_wide":                 # 64 columns: the full line buffer
        return (16, 12, 64), [
            dict(id="pw", kind="conv", Cout=32, K=1, S=1, pad=0, relu=True),
            dict(id="dw", kind="dw",   Cout=32, K=5, S=1, pad=2, relu=True),
        ]
    if case == "k5_deep":                 # 72 channel groups, as EfficientNet-Lite0
        return (192, 7, 7), [
            dict(id="pw", kind="conv", Cout=1152, K=1, S=1, pad=0, relu=True),
            dict(id="dw", kind="dw",   Cout=1152, K=5, S=1, pad=2, relu=True),
        ]
    if case == "k5_block":                # EfficientNet-Lite inverted residual
        return (40, 14, 14), [
            dict(id="expand",  kind="conv", Cout=240, K=1, S=1, pad=0, relu=True),
            dict(id="dw",      kind="dw",   Cout=240, K=5, S=1, pad=2, relu=True),
            dict(id="project", kind="conv", Cout=40,  K=1, S=1, pad=0, relu=False,
                 residual=True, post_add_relu=False),
        ]
    # Wider than one line-buffer row: the fused stage tiles by column.
    if case == "wide_s1":
        return (16, 20, 100), [
            dict(id="pw", kind="conv", Cout=32, K=1, S=1, pad=0, relu=True),
            dict(id="dw", kind="dw",   Cout=32, K=3, S=1, pad=1, relu=True),
        ]
    if case == "wide_s2":                 # MobileNetV2's 112-wide expansion pair
        return (16, 20, 112), [
            dict(id="pw", kind="conv", Cout=48, K=1, S=1, pad=0, relu=True),
            dict(id="dw", kind="dw",   Cout=48, K=3, S=2, pad=1, relu=True),
        ]
    if case == "stem_wide":               # a 3x3 stride-2 stem feeding 112-wide DW
        return (3, 40, 224), [
            dict(id="stem", kind="conv", Cout=32, K=3, S=2, pad=1, relu=True),
            dict(id="dw",   kind="dw",   Cout=32, K=3, S=1, pad=1, relu=True),
        ]
    if case == "k5_wide100":
        return (16, 16, 100), [
            dict(id="pw", kind="conv", Cout=32, K=1, S=1, pad=0, relu=True),
            dict(id="dw", kind="dw",   Cout=32, K=5, S=1, pad=2, relu=True),
        ]
    raise ValueError(f"unknown case '{case}'")


CASES = ["single", "stride2", "ragged", "rowtiled", "after_k3", "v2_block",
         "k5_single", "k5_stride2", "k5_ragged", "k5_rowtiled", "k5_rowtiled_s2",
         "k5_wide", "k5_deep", "k5_block", "wide_s1", "wide_s2", "stem_wide",
         "k5_wide100"]


def gen_case(out_root, case, seed):
    (Cin0, H0, W0), layers = build_layers(case)
    rng = np.random.default_rng(seed + zlib.crc32(case.encode()) % 1000)
    out_dir = os.path.join(out_root, f"pwdw_{case}")
    inputs = os.path.join(out_dir, "inputs")
    params = os.path.join(out_dir, "params")
    refs = os.path.join(out_dir, "refs")
    for d in (inputs, params, refs):
        os.makedirs(d, exist_ok=True)

    ifm0 = rng.integers(-IFM_RANGE, IFM_RANGE + 1, size=(Cin0, H0, W0),
                        dtype=np.int8)
    ifm0.tofile(os.path.join(inputs, "ifm.int8.bin"))

    input_id = "ifm"
    tensors = {input_id: {"kind": "input", "dtype": "int8", "layout": "CHW",
                          "shape": [Cin0, H0, W0], "frac": FRAC_ACT,
                          "file": "inputs/ifm.int8.bin"}}
    nodes = []
    cur = ifm0
    cur_id = input_id
    Cin, H, W = Cin0, H0, W0
    last_out_id = None

    for li, L in enumerate(layers):
        lid = L["id"]
        K, S, pad = L["K"], L["S"], L["pad"]
        Cout = L["Cout"]
        depthwise = L["kind"] == "dw"
        reduction = (K * K) if depthwise else (Cin * K * K)
        Rw = weight_range(reduction)

        if depthwise:
            wshape = (Cout, 1, K, K)
        else:
            wshape = (Cout, Cin, K, K)
        weight = rng.integers(-Rw, Rw + 1, size=wshape, dtype=np.int8)
        bias_i8 = rng.integers(-BIAS_RANGE, BIAS_RANGE + 1, size=(Cout,),
                               dtype=np.int8)
        bias_i32 = shift_bias(bias_i8)

        residual = bool(L.get("residual", False))
        # Residual launches apply any ReLU after the addition, not convolution.
        conv_relu = L["relu"] and not residual
        out, OH, OW = conv_ref(cur, weight, bias_i32, Cin, H, W, Cout, K, S, pad,
                               depthwise, conv_relu)
        post_add_relu = bool(L.get("post_add_relu", False))
        if residual:
            # The residual source must match the projected output exactly.
            assert out.shape == ifm0.shape, "residual shape mismatch"
            out = residual_ref(out, ifm0, post_add_relu=post_add_relu)

        weight.tofile(os.path.join(params, f"{lid}.weight.int8.bin"))
        bias_i8.tofile(os.path.join(params, f"{lid}.bias.int8.bin"))

        w_layout = "OIHW"
        out_id = f"{lid}_out"
        is_last = (li == len(layers) - 1)
        tensors[f"{lid}_w"] = {"kind": "param", "dtype": "int8",
                               "layout": w_layout, "shape": list(wshape),
                               "frac": FRAC_W,
                               "file": f"params/{lid}.weight.int8.bin"}
        tensors[f"{lid}_b"] = {"kind": "param", "dtype": "int8", "layout": "O",
                               "shape": [Cout], "frac": FRAC_B,
                               "file": f"params/{lid}.bias.int8.bin"}
        tensors[out_id] = {"kind": "activation", "dtype": "int8", "layout": "CHW",
                           "shape": [Cout, OH, OW], "frac": FRAC_ACT}

        node = {
            "id": lid, "op": "conv2d",
            "inputs": {"ifm": cur_id, "weight": f"{lid}_w", "bias": f"{lid}_b"},
            "outputs": {"ofm": out_id},
            "attrs": {"kernel": [K, K], "stride": [S, S],
                      "padding": [pad, pad, pad, pad], "dilation": [1, 1],
                      "groups": Cout if depthwise else 1},
        }
        if residual:
            node["inputs"]["residual"] = input_id
            node["post_ops"] = {"residual_add": True,
                                "post_add_relu": post_add_relu}
        else:
            node["post_ops"] = {"relu": bool(L["relu"])}
        nodes.append(node)

        cur, cur_id = out, out_id
        Cin, H, W = Cout, OH, OW
        last_out_id = out_id

    cur.tofile(os.path.join(refs, "out_ref.int8.bin"))
    ref_id = f"{last_out_id}_ref"
    tensors[ref_id] = {"kind": "reference", "dtype": "int8", "layout": "CHW",
                       "shape": [Cin, H, W], "frac": FRAC_ACT,
                       "file": "refs/out_ref.int8.bin"}

    graph = {
        "schema": "deeptile.graph.v1",
        "model": {"name": f"pwdw_{case}", "producer": "gen_pwdw_testcase",
                  "producer_version": "0.1.0", "source_framework": "synthetic"},
        "target": {"bitwidth": 8, "signed": True, "activation_layout": "CHW",
                   "conv_weight_layout": "OIHW", "linear_weight_layout": "OI",
                   "bias_dtype": "int8"},
        "graph": {"inputs": [input_id], "outputs": [last_out_id],
                  "references": {last_out_id: ref_id}},
        "tensors": tensors,
        "nodes": nodes,
    }
    with open(os.path.join(out_dir, "graph.json"), "w") as f:
        json.dump(graph, f, indent=2)

    # Report saturation and value diversity for fixture review.
    flat = cur.reshape(-1)
    sat = np.mean((flat == 127) | (flat == -128))
    zero = np.mean(flat == 0)
    print(f"[gen_pwdw] pwdw_{case}: in {Cin0}x{H0}x{W0} -> out {Cin}x{H}x{W}, "
          f"{len(nodes)} nodes, sat={sat:.1%} zero={zero:.1%} "
          f"distinct={len(np.unique(flat))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root")
    ap.add_argument("--case", default="all")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cases = CASES if a.case == "all" else [a.case]
    for c in cases:
        gen_case(a.out_root, c, a.seed)


if __name__ == "__main__":
    main()
