#!/usr/bin/env python3
"""Plot cost-model operating points for every ResNet-50 layer.

Layer data comes from the scheduler inspection interface:

    build/tools/schedule_inspect --resnet --csv

Recorded board testcases from p0_metrics.py are overlaid for comparison.

Usage:
    python3 scripts/tools/p0_roofline_resnet.py [-o OUT.png] [--csv FILE]
                                                 [--binary PATH] [--show]

Writes build/analysis/p0_roofline_resnet.png
by default. Needs matplotlib.
"""

import argparse
import csv as csvmod
import io
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p0_metrics as m  # noqa: E402  (peak/BW constants + 7 board TCs)

OPS_PER_MAC = 2.0
HP_BW_GBPS = m.HP_BW_MBS / 1000.0
PORTS = (1, 2, 3)

STYLE = {  # bound class -> colour, marker, legend
    "MAC":    ("#2ca02c", "o", "MAC-bound"),
    "Weight": ("#d62728", "s", "Weight-bound"),
    "IFM":    ("#1f77b4", "^", "IFM-bound"),
}

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))


def mem_roof(intensity, ports):
    return OPS_PER_MAC * intensity * HP_BW_GBPS * ports


def load_rows(csv_path, binary):
    """Return the per-layer CSV rows, from a file or by running schedule_inspect."""
    if csv_path:
        with open(csv_path, newline="") as fh:
            return list(csvmod.DictReader(fh))
    if binary is None:
        binary = os.path.join(REPO, "build", "tools", "schedule_inspect")
    if not os.path.exists(binary):
        sys.exit(f"schedule_inspect not found at {binary}\n"
                 "  build it:  cmake --build build/tools --target schedule_inspect\n"
                 "  or pass --csv FILE / --binary PATH")
    out = subprocess.run([binary, "--resnet", "--csv"],
                         capture_output=True, text=True, check=True).stdout
    return list(csvmod.DictReader(io.StringIO(out)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    default_out = os.path.join(REPO, "build", "analysis",
                               "p0_roofline_resnet.png")
    ap.add_argument("-o", "--out", default=default_out)
    ap.add_argument("--csv", help="read a pre-generated --resnet --csv file instead of running the tool")
    ap.add_argument("--binary", help="path to schedule_inspect")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.csv, args.binary)
    if not rows:
        sys.exit("no layer rows parsed")

    try:
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib is required: pip install matplotlib")

    peak = m.PEAK_GOPS

    pts = []  # (intensity, gops, bound, name)
    for r in rows:
        i = float(r["intensity_mac_per_byte"])
        g = float(r["pred_gops"])
        if i > 0 and g > 0:
            pts.append((i, g, r["bound"], r["name"]))

    fig, ax = plt.subplots(figsize=(9.0, 6.2))

    # Cover both predicted layers and measured testcase overlays.
    all_i = [p[0] for p in pts] + [tc.op_intensity for tc in m.TCS]
    all_g = [p[1] for p in pts] + [tc.achieved_gops for tc in m.TCS]
    i_lo, i_hi = min(all_i) * 0.6, max(all_i) * 1.8
    xs = [i_lo * (i_hi / i_lo) ** (k / 400.0) for k in range(401)]

    # Compute and memory roofs.
    ax.axhline(peak, color="black", lw=1.8)
    ax.text(i_hi, peak * 1.04, f"compute roof  {peak:.1f} GOP/s",
            ha="right", va="bottom", fontsize=9, fontweight="bold")
    for ports in PORTS:
        ys = [min(peak, mem_roof(x, ports)) for x in xs]
        ax.plot(xs, ys, color="0.45", lw=1.2, ls={1: "-", 2: "--", 3: ":"}[ports])
        ridge_i = peak / (OPS_PER_MAC * HP_BW_GBPS * ports)
        xi = ridge_i * 0.28
        ax.text(xi, mem_roof(xi, ports) * 1.05,
                f"{ports}×HP {ports * HP_BW_GBPS:.1f} GB/s",
                rotation=37, rotation_mode="anchor", ha="left", va="bottom",
                fontsize=8, color="0.35")

    # Predicted layer operating points.
    seen = set()
    for i, g, bnd, _name in pts:
        colour, marker, legend = STYLE.get(bnd, ("0.5", "x", bnd))
        ax.scatter(i, g, s=34, color=colour, marker=marker, alpha=0.75,
                   edgecolor="none", zorder=4,
                   label=(f"{legend} (predicted)" if bnd not in seen else None))
        seen.add(bnd)

    # Selected layer labels.
    landmarks = {"stem_conv": (6, -11), "fc_gap": (8, 10),
                 "conv5_x_b0_conv3": (10, -14)}
    for i, g, _b, name in pts:
        if name in landmarks:
            ax.annotate(name, (i, g), textcoords="offset points",
                        xytext=landmarks[name], fontsize=7.5, color="0.2")

    # Measured testcase overlays.
    for tc in m.TCS:
        bnd = "IFM" if "IFM" in tc.label else "Weight" if "Weight" in tc.label else "MAC"
        colour, marker, _ = STYLE[bnd]
        ax.scatter(tc.op_intensity, tc.achieved_gops, s=95, color=colour,
                   marker=marker, edgecolor="black", linewidth=1.1, zorder=6,
                   label=("board-measured TC" if "board-measured TC" not in seen else None))
        seen.add("board-measured TC")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(i_lo, i_hi)
    ax.set_ylim(min(all_g) * 0.5, peak * 1.6)
    ax.set_xlabel("Operational intensity  (MAC / DDR byte)")
    ax.set_ylabel("Throughput  (GOP/s)")
    ax.set_title("DeepTile P0 roofline — all 54 ResNet-50 layers (cost-model predicted) @ 200 MHz")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}  ({len(pts)} layers)")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
