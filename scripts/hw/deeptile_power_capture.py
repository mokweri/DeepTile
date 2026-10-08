#!/usr/bin/env python3
"""Measure ZCU104 board energy across a command's execution.

Runs on the board. Captures the 12 V input rail from the INA226 through the IIO
buffer, which timestamps each sample in the hardware driver rather than at the
moment userspace happens to read it, and concurrently polls the two IRPS5401
PMICs for the per-rail decomposition.

Energy is integrated over the wrapped command's own window, so a J/image figure
divides the same interval the host reports throughput over. An idle baseline is
measured before the command so dynamic power can be separated from the board's
static draw (fans, PHY, DDR refresh, and the programmed but quiescent PL).

Requires root: the IIO buffer attributes are not world-writable.
"""

import argparse
import hashlib
import json
import os
import select
import struct
import subprocess
import sys
import threading
import time

IIO = "/sys/bus/iio/devices/iio:device1"
IIO_DEV = "/dev/iio:device1"

# Channels enabled for buffered capture. The driver packs enabled channels in
# scan-index order with natural alignment, so this order fixes the record
# layout; the timestamp is 8-byte aligned, giving a 16-byte stride.
SCAN = ("in_voltage1", "in_power2", "in_current3", "in_timestamp")
STRIDE = 16
OFF_VBUS, OFF_POWER, OFF_CURRENT, OFF_TS = 0, 2, 4, 8


def rd(path, root=IIO):
    with open(os.path.join(root, path)) as f:
        return f.read().strip()


def wr(path, value, root=IIO):
    with open(os.path.join(root, path), "w") as f:
        f.write(str(value))


# Schematic net names for the IRPS5401 outputs, from UG1267 Table 3-31. Keyed
# by PMBus address and hwmon channel so a decomposition names board rails
# rather than channel indices. Addresses, not hwmon indices: hwmon numbering is
# probe-order and shifts across boots.
#
# U179 gangs PHASE_C and PHASE_D onto VCC1V2, so its pout4 has no rail and
# reports nothing; that is the board's design, not a failed read.
IRPS5401_RAILS = {
    "0043": {"pout1": "VCCINT", "pout2": "VCC1V8", "pout3": "VCC1V2",
             "pout5": "MGTAVCC"},
    "0044": {"pout1": "UTIL_3V3", "pout2": "VCC1V13", "pout3": "UTIL_5V0",
             "pout4": "VADJ_FMC", "pout5": "MGTRAVCC"},
}

# Rails the accelerator's own work appears on: the PL core supply and the DDR4
# interface supply. Reported first so a reader sees them before the constants.
WORK_RAILS = ("VCCINT", "VCC1V2")


def rail_name(address, label):
    """Map a PMIC channel to its schematic net, or None when unrecognized.

    Returns None rather than guessing on an unknown board revision: an
    unnamed channel is honest, a wrongly named one is not.
    """
    suffix = address.split("-")[-1] if "-" in address else address
    return IRPS5401_RAILS.get(suffix, {}).get(label)


def discover_pmic_rails():
    """Return [(label, path)] for every IRPS5401 output-power channel.

    hwmon indices are assigned in probe order and are not stable across boots,
    so devices are matched on driver name and I2C address instead.
    """
    rails = []
    base = "/sys/class/hwmon"
    for entry in sorted(os.listdir(base)):
        hw = os.path.join(base, entry)
        try:
            if rd("name", hw) != "irps5401":
                continue
            addr = os.path.basename(os.path.realpath(os.path.join(hw, "device")))
        except OSError:
            continue
        for attr in sorted(os.listdir(hw)):
            if not (attr.startswith("power") and attr.endswith("_input")):
                continue
            stem = attr[: -len("_input")]
            try:
                label = rd(stem + "_label", hw)
            except OSError:
                label = stem
            # pout* is a regulated output rail; pin* is that regulator's draw
            # from the 12 V input the INA226 already measures, so including it
            # in a total would double-count.
            net = rail_name(addr, label)
            rails.append({
                "device": addr,
                "channel": stem,
                "label": label,
                "rail": net,
                "kind": "output" if label.startswith("pout") else "input",
                "path": os.path.join(hw, attr),
            })
    return rails


class RailPoller(threading.Thread):
    """Sample the PMIC rails on a slow cadence alongside the IIO buffer.

    PMBus reads require a page write per channel, so a full sweep is far too
    slow to interleave with the INA226. It runs on its own thread at a rate
    chosen only to characterize the decomposition, not to integrate energy.
    """

    def __init__(self, rails, interval):
        super().__init__(daemon=True)
        self.rails = rails
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            row = {"t": time.clock_gettime(time.CLOCK_REALTIME)}
            for rail in self.rails:
                try:
                    with open(rail["path"]) as f:
                        key = rail["rail"] or (rail["label"] + "@" + rail["device"])
                        row[key] = int(f.read()) / 1e6
                except (OSError, ValueError):
                    pass
            self.samples.append(row)
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


class InaCapture:
    """Buffered INA226 capture with hardware timestamps."""

    def __init__(self, integration_time=None, oversampling=None, length=8192,
                 watermark=16):
        self.length = length
        self.watermark = watermark
        self.integration_time = integration_time
        self.oversampling = oversampling
        self.samples = []
        self._saved = {}
        self._stop = threading.Event()
        self._thread = None
        self.scale = {}

    def __enter__(self):
        for attr in ("in_oversampling_ratio", "in_voltage0_integration_time",
                     "in_voltage1_integration_time", "buffer/length",
                     "buffer/watermark"):
            try:
                self._saved[attr] = rd(attr)
            except OSError:
                pass
        try:
            wr("buffer/enable", 0)
        except OSError:
            pass
        if self.oversampling is not None:
            wr("in_oversampling_ratio", self.oversampling)
        if self.integration_time is not None:
            wr("in_voltage0_integration_time", self.integration_time)
            wr("in_voltage1_integration_time", self.integration_time)
        for name in ("in_power2", "in_voltage1", "in_current3"):
            self.scale[name] = float(rd(name + "_scale"))
        self.rate_hz = float(rd("in_sampling_frequency"))
        self.shunt_ohm = float(rd("in_shunt_resistor"))
        for ch in SCAN:
            wr("scan_elements/%s_en" % ch, 1)
        wr("buffer/length", self.length)
        # Wake once per batch rather than once per sample. The driver
        # timestamps each sample as it is captured, so batching changes only
        # how often userspace is scheduled, never the time base.
        wr("buffer/watermark", self.watermark)
        wr("buffer/enable", 1)
        # Blocking: the reader waits in select() inside the kernel until the
        # watermark is reached. An O_NONBLOCK poll loop instead burns hundreds
        # of syscalls a second, which is invisible on an accelerator-bound
        # workload but measurably slows a host-bound one.
        self._fd = os.open(IIO_DEV, os.O_RDONLY)
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def _reader(self):
        pending = b""
        while not self._stop.is_set():
            # The timeout only bounds how long stopping takes; no data is lost
            # while waiting, because the kernel buffers it.
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.5)
            except (OSError, ValueError):
                break
            if not ready:
                continue
            try:
                chunk = os.read(self._fd, 65536)
            except OSError:
                break
            if not chunk:
                continue
            pending += chunk
            n = len(pending) // STRIDE
            for k in range(n):
                base = k * STRIDE
                self.samples.append((
                    struct.unpack_from("<q", pending, base + OFF_TS)[0],
                    struct.unpack_from("<H", pending, base + OFF_POWER)[0],
                    struct.unpack_from("<H", pending, base + OFF_VBUS)[0],
                    struct.unpack_from("<H", pending, base + OFF_CURRENT)[0],
                ))
            pending = pending[n * STRIDE:]

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            os.close(self._fd)
        except OSError:
            pass
        try:
            wr("buffer/enable", 0)
        except OSError:
            pass
        for ch in SCAN:
            try:
                wr("scan_elements/%s_en" % ch, 0)
            except OSError:
                pass
        for attr, value in self._saved.items():
            try:
                wr(attr, value)
            except OSError:
                pass
        return False

    def watts(self):
        """(timestamp_seconds, watts) pairs, in capture order."""
        s = self.scale["in_power2"] / 1000.0
        return [(ts / 1e9, raw * s) for ts, raw, _, _ in self.samples]


def window_stats(series, t0, t1):
    """Integrate and summarize the samples falling inside [t0, t1]."""
    sel = [(t, w) for t, w in series if t0 <= t <= t1]
    if len(sel) < 2:
        return None
    # Trapezoidal integration on the hardware timestamps: sample spacing is
    # regular but not exact, and gaps must not be assumed uniform.
    energy = 0.0
    for i in range(len(sel) - 1):
        energy += 0.5 * (sel[i][1] + sel[i + 1][1]) * (sel[i + 1][0] - sel[i][0])
    w = [x[1] for x in sel]
    mean = sum(w) / len(w)
    var = sum((x - mean) ** 2 for x in w) / len(w)
    span = sel[-1][0] - sel[0][0]
    return {
        "samples": len(sel),
        "duration_s": span,
        "effective_rate_hz": (len(sel) - 1) / span if span > 0 else 0.0,
        "energy_j": energy,
        "mean_w": mean,
        "sd_w": var ** 0.5,
        "min_w": min(w),
        "max_w": max(w),
        "median_w": sorted(w)[len(w) // 2],
    }


def rail_means(poller, t0, t1):
    rows = [r for r in poller.samples if t0 <= r["t"] <= t1]
    if not rows:
        return {}
    keys = set()
    for r in rows:
        keys.update(k for k in r if k != "t")
    out = {}
    for k in sorted(keys):
        vals = [r[k] for r in rows if k in r]
        if vals:
            out[k] = {"mean_w": sum(vals) / len(vals),
                      "max_w": max(vals), "samples": len(vals)}
    return out


def read_run_record(path):
    """Load a tilecnn.run.v1 record and extract what a power record must cite.

    Returns (info, window) where window is None when the host recorded none.
    A missing window is not an error here: it means the host predates the
    markers, and the caller must fall back and say so in the record.
    """
    # The host's --output names a directory and writes run.json inside it, so
    # accept either form rather than making the caller remember which.
    if os.path.isdir(path):
        path = os.path.join(path, "run.json")
    with open(path) as f:
        record = json.load(f)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    info = {
        "record": os.path.abspath(path),
        "record_sha256": digest,
        "schema": record.get("schema"),
        "mode": record.get("mode"),
        "model": record.get("model"),
        "target": record.get("target"),
        "status": record.get("status"),
    }
    # The count energy is divided by: images for a dataset run, measured runs
    # for a single-graph run. Both are "one inference" for J/inference.
    results = record.get("results", {})
    execution = record.get("execution", {})
    if record.get("mode") == "dataset":
        info["inferences"] = results.get("images", 0)
    else:
        info["inferences"] = execution.get("measured_runs", 0)

    window = record.get("window")
    if not window or "start_ns" not in window or "end_ns" not in window:
        return info, None
    if window.get("clock") != "CLOCK_REALTIME":
        raise ValueError(
            "run record window is on %r; this sampler timestamps on "
            "CLOCK_REALTIME and cannot align to another clock"
            % window.get("clock"))
    return info, (window["start_ns"] / 1e9, window["end_ns"] / 1e9)


def drift_note(before, after):
    """Compare the two idle baselines that bracket a run.

    The fan is on the measured 12 V rail and its speed tracks die temperature,
    so a run that warms the board raises its own baseline. A material gap
    between the two means the dynamic figure carries that uncertainty.
    """
    if not before or not after:
        return None
    delta = after["mean_w"] - before["mean_w"]
    return {
        "before_w": before["mean_w"],
        "after_w": after["mean_w"],
        "delta_w": delta,
        # The before-window's own spread is the natural yardstick: drift within
        # it is indistinguishable from the baseline's normal wander.
        "exceeds_noise": abs(delta) > 3.0 * before["sd_w"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate", choices=("normal", "fast"), default="normal",
                    help="normal: 114 Hz, 8.8 ms averaging window (low noise). "
                         "fast: ~1.5 kHz, resolves inference-scale structure.")
    ap.add_argument("--settle", type=float, default=3.0,
                    help="seconds of idle baseline captured before the command")
    ap.add_argument("--tail", type=float, default=5.0,
                    help="seconds of idle baseline captured after the command; "
                         "compared against the pre-run baseline to expose the "
                         "thermal drift a long run induces in the fan")
    ap.add_argument("--rail-interval", type=float, default=0.0,
                    help="PMIC decomposition polling period in seconds; 0 disables it. "
                         "Each PMBus channel needs a page write, so a full sweep costs "
                         "~28 ms of CPU. Enable it for characterization runs, not for "
                         "runs whose throughput is being reported.")
    ap.add_argument("--run-record",
                    help="tilecnn.run.v1 record the measured command writes. "
                         "Its window bounds the integration, so energy covers "
                         "exactly the interval the host reported throughput "
                         "over. Without it the whole command is integrated, "
                         "which charges startup and weight upload to inference.")
    ap.add_argument("--images", type=int, default=0,
                    help="inference count, when no run record supplies one")
    ap.add_argument("--out", help="write the summary as JSON to this path")
    ap.add_argument("--trace", help="write the raw power trace as CSV to this path")
    ap.add_argument("command", nargs=argparse.REMAINDER,
                    help="-- followed by the command to measure")
    args = ap.parse_args()

    cmd = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not cmd:
        ap.error("no command given; pass it after --")
    if os.geteuid() != 0:
        ap.error("must run as root: the IIO buffer attributes are not world-writable")

    tuning = {"fast": {"integration_time": "0.000332", "oversampling": 1},
              "normal": {"integration_time": "0.001100", "oversampling": 4}}[args.rate]

    rails = discover_pmic_rails() if args.rail_interval > 0 else []
    poller = RailPoller(rails, args.rail_interval) if rails else None

    with InaCapture(**tuning) as ina:
        if poller:
            poller.start()
        idle0_t0 = time.clock_gettime(time.CLOCK_REALTIME)
        time.sleep(args.settle)
        idle0_t1 = time.clock_gettime(time.CLOCK_REALTIME)

        cmd_t0 = time.clock_gettime(time.CLOCK_REALTIME)
        proc = subprocess.run(cmd)
        cmd_t1 = time.clock_gettime(time.CLOCK_REALTIME)

        # A second baseline after the run, not merely a drain interval: the two
        # bracket the run and their difference is the thermal drift.
        time.sleep(args.tail)
        idle1_t1 = time.clock_gettime(time.CLOCK_REALTIME)
        if poller:
            poller.stop()
        series = ina.watts()
        meta = {"nominal_rate_hz": ina.rate_hz, "shunt_ohm": ina.shunt_ohm,
                "power_lsb_w": ina.scale["in_power2"] / 1000.0}

    run_info, host_window = None, None
    if args.run_record:
        try:
            run_info, host_window = read_run_record(args.run_record)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print("error: cannot use run record: %s" % exc, file=sys.stderr)
            return 1
        if host_window is None:
            print("warning: run record carries no window; integrating the whole "
                  "command, which charges startup and weight upload to inference",
                  file=sys.stderr)

    if host_window is not None:
        win_t0, win_t1 = host_window
        window_source = "host"
    else:
        win_t0, win_t1 = cmd_t0, cmd_t1
        window_source = "command"

    idle_before = window_stats(series, idle0_t0, idle0_t1)
    idle_after = window_stats(series, cmd_t1, idle1_t1)
    load = window_stats(series, win_t0, win_t1)
    if load is None:
        print("error: no samples inside the measurement window; the command may "
              "have exited before the sampler saw it", file=sys.stderr)
        return 1

    inferences = args.images
    if run_info and not inferences:
        inferences = run_info.get("inferences", 0)

    dynamic_w = (load["mean_w"] - idle_before["mean_w"]) if idle_before else None
    summary = {
        "schema": "deeptile.power.v1",
        "command": cmd,
        "exit_status": proc.returncode,
        "sensor": dict(meta, rate_mode=args.rate,
                       rail="VCC12",
                       device="ina226 U183 @ 0x40, TCA9548A port 3",
                       scope="whole board: the 12 V input every regulator "
                             "derives from",
                       method="IIO buffered capture, kernel-timestamped"),
        "window": {
            "source": window_source,
            "clock": "CLOCK_REALTIME",
            "start_ns": int(win_t0 * 1e9),
            "end_ns": int(win_t1 * 1e9),
            "duration_s": win_t1 - win_t0,
        },
        "idle_before": idle_before,
        "idle_after": idle_after,
        "baseline_drift": drift_note(idle_before, idle_after),
        "load": load,
        "dynamic_w": dynamic_w,
        "rails": {"idle": rail_means(poller, idle0_t0, idle0_t1),
                  "load": rail_means(poller, win_t0, win_t1)} if poller else {},
    }
    if run_info:
        summary["run"] = run_info
    if inferences > 0:
        summary["inferences"] = inferences
        summary["joules_per_inference"] = load["energy_j"] / inferences
        summary["inferences_per_joule"] = inferences / load["energy_j"]
        if dynamic_w is not None:
            summary["dynamic_joules_per_inference"] = (
                dynamic_w * load["duration_s"] / inferences)

    print("\n=== board energy ===")
    if proc.returncode != 0:
        # Energy over a window whose workload aborted describes nothing. Say so
        # here as well as in the record, so it cannot be read off the terminal
        # and quoted.
        print("  COMMAND FAILED (exit %d) -- the figures below do not describe "
              "a completed run" % proc.returncode)
    print("  sensor      : VCC12 whole-board, %s mode, %.0f Hz nominal, "
          "%.1f Hz achieved, %.1f mW/LSB"
          % (args.rate, meta["nominal_rate_hz"], load["effective_rate_hz"],
             meta["power_lsb_w"] * 1000))
    if window_source == "host":
        print("  window      : host-reported, %.2f s (startup excluded)"
              % load["duration_s"])
    else:
        print("  window      : WHOLE COMMAND, %.2f s -- includes startup and "
              "weight upload" % load["duration_s"])
    if idle_before:
        print("  idle before : %.3f W  (sd %.3f, n=%d)"
              % (idle_before["mean_w"], idle_before["sd_w"],
                 idle_before["samples"]))
    if idle_after:
        print("  idle after  : %.3f W  (sd %.3f, n=%d)"
              % (idle_after["mean_w"], idle_after["sd_w"], idle_after["samples"]))
    drift = summary["baseline_drift"]
    if drift and drift["exceeds_noise"]:
        print("  NOTE        : baseline moved %+.3f W across the run; the board "
              "was not in thermal steady state, and the dynamic figure carries "
              "that uncertainty" % drift["delta_w"])
    print("  under load  : %.3f W  (sd %.3f, n=%d)"
          % (load["mean_w"], load["sd_w"], load["samples"]))
    if dynamic_w is not None:
        print("  dynamic     : %.3f W above idle  (%.1f %% of board power)"
              % (dynamic_w, 100.0 * dynamic_w / load["mean_w"]))
    print("  energy      : %.2f J" % load["energy_j"])
    if inferences > 0:
        print("  per image   : %.4f J total board, %.4f J dynamic  (%.2f img/J)"
              % (summary["joules_per_inference"],
                 summary.get("dynamic_joules_per_inference", float("nan")),
                 summary["inferences_per_joule"]))

    load_rails = summary["rails"].get("load", {})
    if load_rails:
        # Only pout* rails are reported: pin* is each regulator's draw from the
        # 12 V input the INA226 already measures, so showing both invites a sum
        # that double-counts.
        print("\n  PMIC output rails (mean W under load):")
        ordered = ([k for k in WORK_RAILS if k in load_rails] +
                   sorted(k for k in load_rails if k not in WORK_RAILS))
        for k in ordered:
            if k.startswith("pin"):
                continue
            idle_w = summary["rails"]["idle"].get(k, {}).get("mean_w")
            delta = ("  (%+.3f vs idle)" % (load_rails[k]["mean_w"] - idle_w)
                     if idle_w is not None else "")
            mark = " *" if k in WORK_RAILS else ""
            print("    %-14s %7.3f W%s%s"
                  % (k, load_rails[k]["mean_w"], delta, mark))
        print("    * PL core and DDR4 interface: where the accelerator's work "
              "appears.")
        print("    Rails without PMBus telemetry (VCCINT_VCU, VCC3V3, MGT*, "
              "UTIL_2V5, both")
        print("    DDR4_VTT) are visible only inside the 12 V total, so these "
              "will not sum to it.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print("\n  power record -> %s" % args.out)
    if args.trace:
        with open(args.trace, "w") as f:
            f.write("timestamp_s,watts\n")
            for t, w in series:
                f.write("%.9f,%.4f\n" % (t, w))
        print("  trace        -> %s (%d samples)" % (args.trace, len(series)))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
