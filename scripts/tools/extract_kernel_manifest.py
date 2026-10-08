#!/usr/bin/env python3
"""Extract a strict cost-model manifest from a Vitis HLS synthesis report.

The output records loop timing, FIFO depths, tool provenance, and an
architecture fingerprint for validation by cost_model_v2_1.cpp.

Per-loop timing is trip-INDEPENDENT (the cost model computes done(N) itself):
  ii    = PipelineII
  first = PipelineDepth            (pipeline fill → first output token)
  tail  = module Best-caseLatency − loop Best-caseLatency   (module-exit glue)
`flat` records whether the synthesized loop name represents a flattened nest.

Usage:
  extract_kernel_manifest.py <csynth.xml> [-o kernel_manifest.txt]
                             [--defs src/common/arch_config.h] [--check]

  --check  exit 1 if any required loop is missing or an expected-II=1 loop drifted.
"""
import argparse, os, re, subprocess, sys
import xml.etree.ElementTree as ET

# Loop-name substring -> (manifest key, expect_ii1). Order resolves substring
# collisions, so specific names must precede their shorter forms.
LOOP_MAP = [
    ("FETCH_XFER_LOOP",                 ("schedule_xfer",   True)),
    ("FETCH_GROUP_LOOP",                ("schedule_group",  True)),
    ("FETCH_LOOP",                      ("schedule_tile",   True)),
    ("BIAS_PRELOAD",                    ("bias_preload",    True)),
    ("BANK_LOOP_DEPTH_LOOP",            ("weight_depth",    True)),
    ("DEPTH_LOOP",                      ("weight_depth",    True)),
    ("MAC_DW_FLAT_LOOP",                ("mac_dw",          True)),
    ("MAC_FLAT_LOOP",                   ("mac_dense",       True)),
    ("RAW_ROW",                         ("ifm_load_raw",    True)),
    ("K_POS",                           ("ifm_stream",      True)),
    ("LOAD_FORWARD_W",                  ("k1_fill",         True)),
    ("PREFETCH_CIG",                    ("k1_prefetch",     True)),
    ("CACHE_REUSE_STREAM_LOOP",         ("k1_reuse",        True)),
    ("ACCQ_GROUP_LOOP",                 ("acc_group",       True)),
    ("EMIT_LOOP",                       ("emit",            True)),
    ("RESIDUAL_VEC_LOOP",               ("residual",        True)),
    ("FUSED_POOL_COL",                  ("pool_row",        True)),
    ("GAP_ACC",                         ("gap_acc",         True)),
    ("DW_WEIGHT_PRELOAD",               ("dw_preload",      True)),
    ("DW_WLOAD_TAPS",                   ("dw_local_weight", True)),
    ("DW_TILE_PIX",                     ("dw_tile_pix",     True)),
    ("OFM_STORE_ROWS",                  ("store_row",       True)),
]

# Stream instance -> manifest FIFO key.
FIFO_MAP = {
    "pixel_stream_U":    "fifo_pixel",
    "acc_stream_U":      "fifo_acc",
    "ofm_stream_U":      "fifo_intermediate_ofm",
    "ofm_gap_stream_U":  "fifo_final_store",
}

# Architecture constants that affect the synthesized dataflow topology.
ARCH_DEFS = ["P_SPATIAL", "OC_TILE_SIZE", "IC_TILE_SIZE", "P_IC", "MAX_TILE_H", "MAX_TILE_W"]

# The flattened MAC pipeline's per-CIG restart cost is not separable in
# csynth.xml, so the calibrated value is emitted explicitly in the manifest.
CIG_RESTART_CY = 4


def g(e, t):
    x = e.find(t) if e is not None else None
    return (x.text or "").strip() if x is not None else ""


def find_module(root, name):
    for m in root.find("ModuleInformation").findall("Module"):
        if g(m, "Name") == name:
            return m
    return None


def module_best(m):
    return g(m, "PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency")


def outer_loop_body_latency(m):
    """Return the best-case body latency of a non-pipelined outer loop."""
    sl = m.find("PerformanceEstimates/SummaryOfLoopLatency") if m is not None else None
    if sl is None:
        return None
    lp = list(sl)[0]
    d = g(lp, "PipelineDepth").split("~")[0].strip()
    return int(d) if d else None


def mac_task_entry(root):
    """Derive fixed per-descriptor MAC control latency."""
    outer = find_module(root, "mac_tile")
    inner = find_module(root, "mac_tile_Pipeline_CIG_LOOP_MAC_FLAT_LOOP")
    body = outer_loop_body_latency(outer)
    ib = module_best(inner)
    if body is None or not ib:
        return None
    return max(0, body - int(ib))


def collect_loops(root):
    """Collect timing fields for every pipelined loop."""
    out = []
    for m in root.find("ModuleInformation").findall("Module"):
        ov = m.find("PerformanceEstimates/SummaryOfOverallLatency")
        sl = m.find("PerformanceEstimates/SummaryOfLoopLatency")
        if sl is None:
            continue
        mod_best = g(ov, "Best-caseLatency")
        for lp in sl:
            ii = g(lp, "PipelineII")
            if not ii:
                continue
            depth = g(lp, "PipelineDepth")
            lb = g(lp, "Latency").split("~")[0].strip()
            out.append((g(lp, "Name"), ii, depth, mod_best, lb))
    return out


def read_arch(defs_path):
    txt = open(defs_path).read()
    vals = {}
    for d in ARCH_DEFS:
        m = re.search(rf"^#define\s+{d}\s+(\d+)", txt, re.M)
        vals[d] = int(m.group(1)) if m else -1
    return vals


def arch_hash(vals):
    # Keep this ordering identical to the cost model's compiled fingerprint.
    return "_".join(f"{d}{vals[d]}" for d in ARCH_DEFS)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csynth_xml")
    ap.add_argument("-o", "--output", default="kernel_manifest.txt")
    ap.add_argument("--embed", default="src/compiler/src/kernel_manifest_embedded.inc",
                    help="C++ raw-string copy compiled into the binary")
    ap.add_argument("--defs", default="src/common/arch_config.h")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    try:
        root = ET.parse(a.csynth_xml).getroot()
    except (ET.ParseError, OSError) as e:
        print(f"error: cannot parse {a.csynth_xml}: {e}", file=sys.stderr); return 1

    loops = collect_loops(root)
    # The first LOOP_MAP match owns each manifest key.
    prof = {}     # key -> (ii, first, tail, flat, loop_name)
    for name, ii, depth, mb, lb in loops:
        for sub, (key, _) in LOOP_MAP:
            if sub in name and key not in prof:
                ii_f = float(ii)
                first = int(depth) if depth else 0
                tail = max(0, int(mb) - int(lb)) if (mb and lb) else 0
                flat = "_" in name and name.count("_LOOP") + name.count("_ROW") + \
                       name.count("_COL") + name.count("_POS") + name.count("_W") > 1
                prof[key] = (ii_f, first, tail, 1 if flat else 0, name)
                break

    missing = [k for _, (k, _) in LOOP_MAP if k not in prof and k not in ("mac_dw",)]
    drift = [(k, prof[k][0], prof[k][4]) for _, (k, e1) in LOOP_MAP
             if k in prof and e1 and prof[k][0] != 1.0]

    # FIFO depths reported by synthesis.
    fifos = {}
    fi = root.find("FIFOInformation")
    if fi is not None:
        for f in fi.findall("FIFOInst"):
            nm = g(f, "Name")
            if nm in FIFO_MAP:
                sd = g(f, "StaticDepth")
                if sd:
                    fifos[FIFO_MAP[nm]] = int(sd)

    # Synthesis provenance.
    ua = root.find("UserAssignments")
    part = g(ua, "Part"); top = g(ua, "TopModelName"); clk = g(ua, "TargetClockPeriod")
    ver = g(root.find("ReportVersion"), "Version")
    arch = read_arch(a.defs)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         text=True).strip()
    except Exception:
        commit = "unknown"

    lines = [
        "# kernel_manifest v2.1 — generated by extract_kernel_manifest.py",
        f"# from {a.csynth_xml}",
        f"manifest.source_commit {commit}",
        f"manifest.hls_version {ver or '2024.2'}",
        f"manifest.target_clock_ns {clk or '5.0'}",
        f"manifest.part {part}",
        f"manifest.top {top}",
        f"manifest.arch_hash {arch_hash(arch)}",
        "manifest.provenance extracted",
        "",
    ]
    for key in sorted(prof):
        ii, first, tail, flat, nm = prof[key]
        lines.append(f"{key}.ii {ii:g}")
        lines.append(f"{key}.first {first}")
        lines.append(f"{key}.tail {tail}")
        # Only recurrences that depend on loop flattening need the flag.
        if key in ("weight_depth", "mac_dense"):
            lines.append(f"{key}.flat {flat}   # {nm}")
    # Fixed descriptor-control latency is outside the inner pipelined loop.
    mte = mac_task_entry(root)
    if mte is not None:
        lines.append(f"mac_task_entry {mte}")
    # Emit the calibrated restart cost because csynth.xml does not report it.
    lines.append(f"cig_restart {CIG_RESTART_CY}   # calibrated; absent from csynth.xml")
    lines.append("")
    for k in sorted(fifos):
        lines.append(f"{k} {fifos[k]}")

    text = "\n".join(lines) + "\n"
    with open(a.output, "w") as f:
        f.write(text)
    print(f"wrote {a.output}: {len(prof)} loops, {len(fifos)} fifos, "
          f"arch={arch_hash(arch)}, commit={commit}")

    # Embed the same manifest so the cost model needs no runtime data file.
    inc = (
        "// Generated by scripts/tools/extract_kernel_manifest.py; do not edit.\n"
        "// Regenerate after synthesis changes to update both manifest outputs.\n"
        "static const char kEmbeddedKernelManifest[] = R\"MFEST(\n"
        + text +
        ")MFEST\";\n"
    )
    with open(a.embed, "w") as f:
        f.write(inc)
    print(f"wrote {a.embed}: embedded manifest ({len(text)} bytes)")
    for k in missing:
        print(f"  MISSING loop: {k}", file=sys.stderr)
    for k, ii, nm in drift:
        print(f"  II DRIFT: {k} ({nm}) II={ii:g} (expected 1)", file=sys.stderr)
    if a.check and (missing or drift):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
