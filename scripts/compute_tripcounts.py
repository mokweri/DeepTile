#!/usr/bin/env python3
"""Emit per-graph HLS trip-count and AXI-depth definitions.

Runs graph_plan and writes one shell-ready line of -DTC_* and -DAXI_DEPTH_*
flags to stdout. The HLS Tcl flow consumes these definitions before registering
the design sources.
"""

import math
import os
import re
import subprocess
import sys


REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def find_tool(build_dir):
    candidates = [
        os.path.join(build_dir, "graph_plan"),
        os.path.join(REPO_ROOT, "build", "tools", "graph_plan"),
    ]
    for p in candidates:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return os.path.normpath(p)
    return None


def print_build_instructions(build_dir):
    # Vitis may place an incomplete host GCC first on PATH, so build with the
    # system compiler from a normal terminal.
    print(
        f"[compute_tripcounts] graph_plan not found.\n"
        f"  Build it once from a system terminal (outside Vitis):\n"
        f"    cmake -S {os.path.join(REPO_ROOT, 'scripts')} -B {os.path.abspath(build_dir)} -DCMAKE_CXX_COMPILER=/usr/bin/g++\n"
        f"    cmake --build {os.path.abspath(build_dir)} --target graph_plan",
        file=sys.stderr
    )
    sys.exit(1)


def run_tool(tool, graph_json):
    result = subprocess.run([tool, graph_json], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[compute_tripcounts] graph_plan failed:\n{result.stderr}",
              file=sys.stderr)
        sys.exit(1)
    return result.stdout


def parse_one_launch(block):
    """Parse one graph_plan launch block."""
    shape = re.search(
        r"shape:\s*(\d+)x(\d+)x(\d+)\s*->\s*(\d+)x(\d+)x(\d+)"
        r"\s+k=(\d+)x\d+\s+stride=(\d+)",
        block
    )
    tile  = re.search(r"tile:\s*T_oh=(\d+)\s+T_ow=(\d+)", block)
    sched = re.search(r"schedule:\s*transfers=(\d+)\s+compute_tiles=(\d+)", block)
    if not (shape and tile and sched):
        return None

    post_m = re.search(r"post_ops:\s*(.*)", block)
    post_str = post_m.group(1) if post_m else ""
    has_gap      = "gap"     in post_str
    has_maxpool  = "maxpool" in post_str
    has_residual = bool(re.search(r"\bresidual=\w", block))

    # Cache and chaining choices cannot be inferred from tensor geometry. Missing
    # fields retain compatibility with older graph_plan output.
    cache_m = re.search(r"ifm_cache=(on|off)", block)
    chain_m = re.search(r"cig_chain=(on|off)", block)
    uses_ifm_cache = (cache_m.group(1) == "on") if cache_m else None
    uses_cig_chain = (chain_m.group(1) == "on") if chain_m else None

    return dict(
        cin=int(shape.group(1)), ifm_h=int(shape.group(2)), ifm_w=int(shape.group(3)),
        cout=int(shape.group(4)), conv_h=int(shape.group(5)), conv_w=int(shape.group(6)),
        k=int(shape.group(7)),   s=int(shape.group(8)),
        t_oh=int(tile.group(1)), t_ow=int(tile.group(2)),
        n_xfers=int(sched.group(1)), num_tiles=int(sched.group(2)),
        has_gap=has_gap, has_maxpool=has_maxpool, has_residual=has_residual,
        uses_ifm_cache=uses_ifm_cache, uses_cig_chain=uses_cig_chain,
    )


def read_arch_define(name):
    """Read an integer architecture definition from arch_config.h."""
    path = os.path.join(REPO_ROOT, "src", "common", "arch_config.h")
    with open(path) as fh:
        m = re.search(rf"^#define\s+{name}\s+(\d+)", fh.read(), re.M)
    if m is None:
        sys.exit(f"error: could not read #define {name} from {path}")
    return int(m.group(1))


P_SPATIAL = read_arch_define("P_SPATIAL")
TILE = read_arch_define("IC_TILE_SIZE")
if TILE != read_arch_define("OC_TILE_SIZE"):
    sys.exit("error: compute_tripcounts.py assumes IC_TILE_SIZE == OC_TILE_SIZE; "
             "split the two uses of TILE below before diverging them")


def compute_launch_flags(lv):
    """Compute TC_* and AXI_DEPTH_* for one launch."""

    cin   = lv["cin"];  ifm_h = lv["ifm_h"]; ifm_w = lv["ifm_w"]
    cout  = lv["cout"]; conv_h = lv["conv_h"]; conv_w = lv["conv_w"]
    k = lv["k"]; s = lv["s"]
    t_oh = lv["t_oh"]; t_ow = lv["t_ow"]
    n_xfers = lv["n_xfers"]; num_tiles = lv["num_tiles"]

    cig = math.ceil(cin  / TILE)
    cog = math.ceil(cout / TILE)
    k2  = k * k

    # Post-operations determine the final output allocation depth.
    if lv["has_gap"]:
        ofm_h, ofm_w = 1, 1
    elif lv["has_maxpool"]:
        ofm_h = math.ceil(conv_h / 2)
        ofm_w = math.ceil(conv_w / 2)
    else:
        ofm_h, ofm_w = conv_h, conv_w

    # Chained schedules emit one descriptor per COG; unchained schedules emit one
    # per (COG, CIG). Divisibility is only a fallback for older tool output.
    chained = lv.get("uses_cig_chain")
    if chained is not None:
        per_spatial = cog if chained else cig * cog
        num_spatial = num_tiles // per_spatial if per_spatial > 0 else 1
    elif cig * cog > 0 and num_tiles % (cig * cog) == 0:
        num_spatial = num_tiles // (cig * cog)
    elif cog > 0 and num_tiles % cog == 0:
        num_spatial = num_tiles // cog
    else:
        num_spatial = 1

    # INVARIANT: Grouping must match layer_row_aligned in the kernel scheduler.
    # Cached K=1 data is flat-grouped; all other producers preserve row boundaries.
    cached = lv.get("uses_ifm_cache")
    flat_grouping = (k == 1) if cached is None else (k == 1 and cached)
    if flat_grouping:
        num_groups = math.ceil(t_oh * t_ow / P_SPATIAL)
    else:
        num_groups = t_oh * math.ceil(t_ow / P_SPATIAL)
    spatial     = t_oh * t_ow
    depth_count = cig * k2
    compute_cnt = num_tiles // n_xfers if n_xfers > 0 else num_tiles
    padded_h    = (t_oh - 1) * s + k
    padded_w    = (t_ow - 1) * s + k
    rowlen      = (t_ow - 1) * s + 1
    ofm_wb      = cog * num_spatial
    mac_flat    = num_groups * k2
    ifm_pp      = num_tiles - 1
    ofm_pp      = max(0, ofm_wb - 1)

    # Co-simulation AXI depths are element counts, not bytes.
    axi_ifm      = cig * ifm_h * ifm_w           # pixel_vector_dt elements
    axi_weights  = TILE * cog * cig * k2          # weight_vector_dt elements
    axi_biases   = cog                            # biases_vector_dt elements (one per COG)
    axi_out      = cog * ofm_h * ofm_w            # post-GAP/pool output shape
    # Residual input uses the pre-GAP convolution shape.
    axi_residual = (cog * conv_h * conv_w) if lv["has_residual"] else 1
    axi_xdesc    = n_xfers
    axi_cdesc    = num_tiles

    return {
        "TC_K2":            k2,
        "TC_CIG":           cig,
        "TC_COG":           cog,
        "TC_T_OH":          t_oh,
        "TC_T_OW":          t_ow,
        "TC_NUM_SPATIAL":   num_spatial,
        "TC_NUM_XFERS":     n_xfers,
        "TC_NUM_TILES":     num_tiles,
        "TC_NUM_GROUPS":    num_groups,
        "TC_SPATIAL":       spatial,
        "TC_DEPTH_COUNT":   depth_count,
        "TC_COMPUTE_COUNT": compute_cnt,
        "TC_PADDED_H":      padded_h,
        "TC_PADDED_W":      padded_w,
        "TC_ROWLEN":        rowlen,
        "TC_OFM_WRITEBACK": ofm_wb,
        "TC_NUM_BIAS_VECS": cog,
        "TC_PP_LOOP":       n_xfers - 1,
        "TC_MAC_FLAT":      mac_flat,
        "TC_IFM_PP":        ifm_pp,
        "TC_OFM_PP":        ofm_pp,
        "AXI_DEPTH_IFM":      axi_ifm,
        "AXI_DEPTH_WEIGHTS":  axi_weights,
        "AXI_DEPTH_BIASES":   axi_biases,
        "AXI_DEPTH_OUT":      axi_out,
        "AXI_DEPTH_RESIDUAL": axi_residual,
        "AXI_DEPTH_XDESC":    axi_xdesc,
        "AXI_DEPTH_CDESC":    axi_cdesc,
    }


def parse_output(text):
    blocks = re.split(r"(?=Launch \d+:)", text)
    launches = [parse_one_launch(b) for b in blocks if b.strip().startswith("Launch")]
    launches = [lv for lv in launches if lv is not None]

    if not launches:
        print("[compute_tripcounts] Could not parse graph_plan output.",
              file=sys.stderr)
        print(text, file=sys.stderr)
        sys.exit(1)

    # One set of synthesis bounds must cover every launch in the graph.
    all_flags = [compute_launch_flags(lv) for lv in launches]
    merged = {}
    for flags in all_flags:
        for k, v in flags.items():
            merged[k] = max(merged.get(k, 0), v)
    return merged


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("graph_json")
    p.add_argument("--build-dir", default=os.path.join(REPO_ROOT, "build", "tools"),
                   help="directory containing (or for building) graph_plan")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="print human-readable table to stderr")
    args = p.parse_args()

    tool = find_tool(args.build_dir)
    if tool is None:
        print_build_instructions(args.build_dir)

    raw = run_tool(tool, args.graph_json)
    tc  = parse_output(raw)

    if args.verbose:
        print(f"[compute_tripcounts] {os.path.dirname(args.graph_json)}", file=sys.stderr)
        w = max(len(k) for k in tc)
        for k, v in tc.items():
            print(f"  {k:<{w}} = {v}", file=sys.stderr)

    print(" ".join(f"-D{k}={v}" for k, v in tc.items()))


if __name__ == "__main__":
    main()
