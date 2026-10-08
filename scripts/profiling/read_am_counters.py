#!/usr/bin/env python3
"""Read the study's accelerator monitor after a completed profiling process.

No control/reset register writes. Reading SAMPLE latches the metric counters.
Register definitions and latch semantics match board XRT commit
d05b18dc38cc6804ecb4b3dbe6de23f158319567: core/include/xdp/am.h and
xdp/profile/device/am.cpp. Only the two already-built study targets are allowed.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


ALLOWED_XCLBINS = {
    "b8f0def11b4a35f8a4a1d06bd4218319300ec4279bee15571926536483853852",
    "e753b150b05e70f039464e611c25226edfeef3294e3e13349ba7b81461a3fb06",
}
# Both routed debug_ip_layout.rtd files are byte-identical; no arbitrary MMIO.
BASE = 0x80040000
REGISTERS = {
    "execution_count": (0x80, 0xA0),
    "execution_cycles": (0x84, 0xA4),
    "stall_internal_cycles": (0x88, 0xA8),
    "stall_stream_cycles": (0x8C, 0xAC),
    "stall_external_cycles": (0x90, 0xB0),
    "min_execution_cycles": (0x94, 0xB4),
    "max_execution_cycles": (0x98, 0xB8),
    "total_starts": (0x9C, 0xBC),
    "busy_cycles": (0xC0, 0xC4),
    "max_parallel_iterations": (0xC8, 0xCC),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_counters(read32):
    sample_interval = read32(0x20)
    registers = {}
    counters = {}
    for name, (lower, upper) in REGISTERS.items():
        lo, hi = read32(lower), read32(upper)
        if not (0 <= lo <= 0xFFFFFFFF and 0 <= hi <= 0xFFFFFFFF):
            raise ValueError("Invalid 32-bit register value")
        registers[hex(lower)], registers[hex(upper)] = lo, hi
        counters[name] = lo | (hi << 32)
    return sample_interval, registers, counters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("root needed for the board's read-only devmem accesses")
    if args.output.exists() or args.expected_count < 1:
        parser.error("new output and positive execution count required")
    target_path = args.target.resolve()
    target = json.loads(target_path.read_text())
    xclbin = target_path.parent / target["artifact"]["xclbin"]
    identity = target["artifact"]["sha256"]
    if identity not in ALLOWED_XCLBINS or digest(xclbin) != identity:
        raise SystemExit("Not a verified study profile target")
    if digest(target_path.parent / target["boot"]["image"]) != target["boot"]["sha256"]:
        raise SystemExit("Boot identity mismatch")
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    deployed = (target_path.parent / ".deeptile-deploy-boot-id").read_text().strip()
    if not deployed or boot_id == deployed:
        raise SystemExit("Target has not crossed a boot boundary")

    def read32(offset):
        # With no value argument BusyBox devmem only reads a 32-bit register.
        return int(subprocess.check_output(
            ["devmem", hex(BASE + offset), "32"], text=True).strip(), 16)

    interval, registers, counters = sample_counters(read32)
    result = {
        "schema": "deeptile.architecture-am-counters.v1",
        "target": target["target"]["id"], "xclbin_sha256": identity,
        "boot_sha256": target["boot"]["sha256"], "boot_id": boot_id,
        "reader_sha256": digest(Path(__file__).resolve()),
        "base_address": hex(BASE), "sample_interval": interval,
        "registers": registers, "counters": counters,
        "expected_execution_count": args.expected_count,
        "count_matches": counters["execution_count"] == args.expected_count,
        "scope": "entire host process, including warmups; no timed-region subtraction",
        "note": "Execution, busy and stall counters are not additive partitions",
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(counters), flush=True)
    if not result["count_matches"]:
        raise SystemExit("Execution count does not match the completed process")


if __name__ == "__main__":
    main()
