#!/usr/bin/env python3
"""Plot measured P0 testcase operating points against DeepTile rooflines.

Coordinates and architectural limits come from p0_metrics.py.

Usage:
    python3 scripts/tools/p0_roofline_plot.py [-o OUT.png] [--show]

Writes build/analysis/p0_roofline.png by
default. Needs matplotlib.
"""

import argparse
import os
import sys

# Reuse the metric calculations and architecture limits directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p0_metrics as m  # noqa: E402

OPS_PER_MAC = 2.0
HP_BW_GBPS = m.HP_BW_MBS / 1000.0          # 3.2 GB/s per HP port
PORTS = (1, 2, 3)                          # 1-port (weight/ifm) … 3-port (concurrent)

# Plot style by limiting resource.
STYLE = {
    "MAC":    ("#2ca02c", "o", "MAC-bound"),
    "Weight": ("#d62728", "s", "Weight-bound"),
    "IFM":    ("#1f77b4", "^", "IFM-bound"),
}

# Per-point label offsets in points.
LABEL_OFFSET = {
    "TC1": (6, 5), "TC2": (6, 5), "TC4": (6, 5), "TC7f": (8, 2),
    "TC6": (-26, 9), "TC5": (4, 9), "TC3": (6, -13), "TC7c": (9, -5),
}


def bound_of(tc):
    """Measured bound-by class, from the p0_metrics label."""
    label = tc.label
    if "IFM" in label:
        return "IFM"
    if "Weight" in label:
        return "Weight"
    return "MAC"


def mem_roof(intensity, ports):
    """GOP/s ceiling from `ports` HP ports at intensity I (MAC/byte)."""
    return OPS_PER_MAC * intensity * HP_BW_GBPS * ports


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    default_out = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "build",
                     "analysis", "p0_roofline.png"))
    ap.add_argument("-o", "--out", default=default_out, help="output image path")
    ap.add_argument("--show", action="store_true", help="open an interactive window")
    args = ap.parse_args()

    try:
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib is required: pip install matplotlib")

    peak = m.PEAK_GOPS

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    # Logarithmic intensity range with margin around all testcases.
    i_vals = [tc.op_intensity for tc in m.TCS]
    i_lo, i_hi = min(i_vals) * 0.5, max(i_vals) * 2.5
    xs = [i_lo * (i_hi / i_lo) ** (k / 400.0) for k in range(401)]

    # Compute roof.
    ax.axhline(peak, color="black", lw=1.8, ls="-")
    ax.text(i_hi, peak * 1.04, f"compute roof  {peak:.1f} GOP/s",
            ha="right", va="bottom", fontsize=9, fontweight="bold")

    # Memory roofs for concurrent HP ports.
    for ports in PORTS:
        ys = [min(peak, mem_roof(x, ports)) for x in xs]
        ax.plot(xs, ys, color="0.45", lw=1.2,
                ls={1: "-", 2: "--", 3: ":"}[ports])
        # Place each label before its memory roof reaches the compute ceiling.
        ridge_i = peak / (OPS_PER_MAC * HP_BW_GBPS * ports)
        xi = ridge_i * 0.30
        ax.text(xi, mem_roof(xi, ports) * 1.05,
                f"{ports}×HP  {ports * HP_BW_GBPS:.1f} GB/s",
                rotation=37, rotation_mode="anchor",
                ha="left", va="bottom", fontsize=8, color="0.35")

    # Testcase operating points.
    seen = set()
    for tc in m.TCS:
        cls = bound_of(tc)
        colour, marker, legend = STYLE[cls]
        ax.scatter(tc.op_intensity, tc.achieved_gops, s=70, color=colour,
                   marker=marker, edgecolor="black", linewidth=0.6, zorder=5,
                   label=(legend if cls not in seen else None))
        seen.add(cls)
        ax.annotate(tc.name, (tc.op_intensity, tc.achieved_gops),
                    textcoords="offset points",
                    xytext=LABEL_OFFSET.get(tc.name, (6, 5)),
                    fontsize=8, fontweight="bold")

    # Axes and labels.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(i_lo, i_hi)
    ax.set_ylim(min(tc.achieved_gops for tc in m.TCS) * 0.4, peak * 1.6)
    ax.set_xlabel("Operational intensity  (MAC / DDR byte)")
    ax.set_ylabel("Throughput  (GOP/s)")
    ax.set_title("DeepTile P0 baseline roofline — ResNet-50 test cases @ 200 MHz")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
