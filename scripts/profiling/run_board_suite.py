#!/usr/bin/env python3
"""Collect repeated, matched batch-one measurements on an activated target."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_optional(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def read_device_counters(directory, tag):
    # A diagnostic XRT process must not start its own profiler and reset the
    # very counters being inspected. Its configuration is separate from the
    # explicitly selected configuration of the measured host process.
    config = directory / "counter-read.ini"
    if not config.exists():
        config.write_text("[Debug]\nnative_xrt_trace=false\ndevice_trace=off\n"
                          "device_counters=false\nstall_trace=off\n")
    env = dict(os.environ, XRT_INI_PATH=str(config))
    with (directory / f"counter-read-{tag}.log").open("x") as log:
        subprocess.run([
            "xrt-smi", "--batch", "examine", "--report", "debug-ip-status",
            "--format", "JSON", "--output", str(directory / f"device-counters-{tag}.json")],
            stdout=log, stderr=subprocess.STDOUT, env=env, cwd=directory, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--target", type=Path, default=Path("target.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--power-script", type=Path)
    parser.add_argument("--graphs", type=Path, nargs="+",
                        help="Optional isolated graphs; use unique parent-directory names")
    parser.add_argument("--capture-profile", action="store_true",
                        help="Run each process in its own XRT trace directory")
    parser.add_argument("--profile-config", type=Path,
                        help="Override the measured process's xrt.ini (requires --capture-profile)")
    parser.add_argument("--counter-reader", type=Path,
                        help="Optional read-only AM reader; requires profile and sudo -v")
    args = parser.parse_args()
    if min(args.runs, args.repeats) < 1 or args.warmup < 1:
        parser.error("positive runs/repeats and at least one warmup required")
    if args.cpu not in os.sched_getaffinity(0):
        parser.error("selected CPU is unavailable")
    if args.profile_config and not args.capture_profile:
        parser.error("--profile-config requires --capture-profile")
    if args.counter_reader and not args.capture_profile:
        parser.error("--counter-reader requires --capture-profile")
    args.host = args.host.resolve()
    args.target = args.target.resolve()
    args.output = args.output.resolve()
    if args.power_script:
        args.power_script = args.power_script.resolve()
    if args.counter_reader:
        args.counter_reader = args.counter_reader.resolve()
    target = json.loads(args.target.read_text())
    boot = args.target.parent / target["boot"]["image"]
    boot_id = read_optional("/proc/sys/kernel/random/boot_id")
    deploy_id = read_optional(args.target.parent / ".deeptile-deploy-boot-id")
    if not boot_id or not deploy_id or boot_id == deploy_id:
        raise SystemExit("Target activation across a boot boundary is unproven")
    if digest(boot) != target["boot"]["sha256"]:
        raise SystemExit("Boot image differs from target manifest")
    xclbin = args.target.parent / target["artifact"]["xclbin"]
    if digest(xclbin) != target["artifact"]["sha256"]:
        raise SystemExit("XCLBIN differs from target manifest")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("DEEPTILE_", "TILECNN_"))}
    env["DEEPTILE_FUSE_DW"] = "1"
    profile_config = (args.profile_config.resolve() if args.profile_config else
                      args.target.parent / "xrt.ini")
    # Bind both traced and ordinary runs to the configuration recorded below,
    # rather than accidentally inheriting a caller's XRT_INI_PATH.
    env["XRT_INI_PATH"] = str(profile_config)
    if args.capture_profile:
        if not profile_config.is_file():
            parser.error(f"missing profiling configuration: {profile_config}")
    cases = ([(graph.parent.name, graph) for graph in args.graphs] if args.graphs else
             [(model, Path("models") / model / "imagenet1k" / profile / "v1.0.0/graph.json")
              for model, profile in [("mobilenet_v2", "int8-tqt-cle"), ("resnet50", "int8-tqt"),
                                     ("resnet18", "int8-tqt"), ("vgg16_tilecnn", "int8-tqt")]])
    if len({name for name, _ in cases}) != len(cases):
        parser.error("graph parent-directory names must be unique")
    for _, graph in cases:
        if not graph.is_file():
            parser.error(f"missing graph: {graph}")
    args.output.mkdir(parents=True, exist_ok=False)
    context = {
        "schema": "deeptile.architecture-board-suite.v1",
        "target": target,
        "host_sha256": digest(args.host),
        "runner_sha256": digest(Path(__file__).resolve()),
        "power_script_sha256": digest(args.power_script) if args.power_script else None,
        "counter_reader_sha256": digest(args.counter_reader) if args.counter_reader else None,
        "boot_id": boot_id,
        "deploy_boot_id": deploy_id,
        "uname": list(platform.uname()),
        "cpu": args.cpu,
        "frequency_khz": read_optional(
            f"/sys/devices/system/cpu/cpu{args.cpu}/cpufreq/scaling_cur_freq"),
        "governor": read_optional(
            f"/sys/devices/system/cpu/cpu{args.cpu}/cpufreq/scaling_governor"),
        "runs": args.runs, "warmup": args.warmup, "repeats": args.repeats,
        "environment": {"DEEPTILE_FUSE_DW": "1"},
        "capture_profile": args.capture_profile,
        "xrt_ini": read_optional(profile_config),
        "xrt_ini_path": str(profile_config),
        "xrt_ini_sha256": digest(profile_config) if profile_config.is_file() else None,
        "timing_scope": "prepacked_input_upload_through_packed_output_download",
        "excluded": ["model preparation", "image decoding/preprocessing",
                     "output unpacking/reference comparison"],
        "commands": [],
    }
    for repeat in range(args.repeats):
        for model, graph in cases:
            output = args.output / f"{model}-{repeat}"
            workdir = None
            if args.capture_profile:
                workdir = args.output / f"{model}-{repeat}-trace"
                workdir.mkdir()
                read_device_counters(workdir, "before")
            command = ["taskset", "-c", str(args.cpu), str(args.host),
                       str(args.target), str(graph.resolve()), "--measure-latency",
                       "--runs", str(args.runs), "--warmup", str(args.warmup),
                       "--output", str(output)]
            if args.power_script:
                command = ["sudo", "-n", "--preserve-env=DEEPTILE_FUSE_DW,XRT_INI_PATH",
                           "python3", str(args.power_script.resolve()),
                           "--rate", "normal", "--settle", "2", "--tail", "2",
                           "--run-record", str(output),
                           "--out", str(args.output / f"{model}-{repeat}-power.json"),
                           "--trace", str(args.output / f"{model}-{repeat}-power.csv"),
                           "--"] + command
            context["commands"].append(command)
            (args.output / "context.json").write_text(
                json.dumps(context, indent=2) + "\n")
            with (args.output / f"{model}-{repeat}.log").open("x") as log:
                subprocess.run(command, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True, cwd=workdir)
            record = json.loads((output / "run.json").read_text())
            if not record["results"].get("exact_reference_match"):
                raise SystemExit(f"Reference check failed: {output}")
            if args.capture_profile:
                # Retain XRT diagnostics, including their known edge mapping
                # failure. An exit-zero diagnostic is NOT proof of counters.
                # Both captures are outside every timed/power window.
                read_device_counters(workdir, "after")
                if args.counter_reader:
                    expected = ((args.runs + args.warmup) *
                                record["execution"]["launches"])
                    subprocess.run([
                        "sudo", "-n", "python3", str(args.counter_reader),
                        "--target", str(args.target), "--expected-count", str(expected),
                        "--output", str(workdir / "am-counters.json")], check=True)
            print(model, repeat, record["results"], flush=True)


if __name__ == "__main__":
    main()
