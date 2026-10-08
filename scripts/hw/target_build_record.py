#!/usr/bin/env python3
"""Capture and finalize provenance for one DeepTile FPGA build."""

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ARCH_DEFINES = (
    "BITWIDTH",
    "OC_TILE_SIZE",
    "IC_TILE_SIZE",
    "P_IC",
    "P_SPATIAL",
    "MAX_TILE_H",
    "MAX_TILE_W",
    "K_MAX",
    "GAP_SCALE_FRAC_BITS",
    "IFM_CACHE_DEPTH",
    "WEIGHT_BUF_DEPTH",
    "BIAS_BUF_DEPTH",
)

ABI_DEFINES = (
    "DEEPTILE_KERNEL_ABI_VERSION",
    "DEEPTILE_KERNEL_ARGUMENT_COUNT",
    "DEEPTILE_COMPUTE_DESCRIPTOR_BYTES",
    "DEEPTILE_TRANSFER_DESCRIPTOR_BYTES",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_integer_defines(path: Path, names) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    values = {}
    for name in names:
        match = re.search(
            rf"^#define\s+{name}\s+(\d+)\s*(?://.*)?$", text, re.M
        )
        if match is None:
            raise ValueError(f"missing integer define {name} in {path}")
        values[name] = int(match.group(1))
    return values


def read_string_define(path: Path, name: str) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf'^#define\s+{name}\s+"([^"]+)"\s*$', text, re.M)
    if match is None:
        raise ValueError(f"missing string define {name} in {path}")
    return match.group(1)


def atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def repository_state(repo_root: Path) -> tuple[str, bool]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo_root, text=True
    ).strip())
    return revision, dirty


def vitis_version(override: str | None) -> str:
    if override:
        return override
    output = subprocess.check_output(
        ["v++", "--version"], text=True, stderr=subprocess.STDOUT
    )
    match = re.search(r"v(\d{4}\.\d+)", output)
    return match.group(1) if match else "unknown"


def parse_settings(values: list[str]) -> dict[str, str]:
    settings = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"build setting must be key=value: {value}")
        key, setting = value.split("=", 1)
        if not key or not setting or key in settings:
            raise ValueError(f"invalid or repeated build setting: {value}")
        settings[key] = setting
    return settings


def capture(args) -> dict:
    repo_root = args.repo_root.resolve()
    revision, dirty = repository_state(repo_root)
    inputs = []
    seen = set()
    for raw_path in args.input:
        path = raw_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"build input not found: {path}")
        try:
            name = path.relative_to(repo_root).as_posix()
        except ValueError:
            name = str(path)
        if name in seen:
            continue
        seen.add(name)
        inputs.append({"path": name, "sha256": sha256_file(path)})

    return {
        "schema": "deeptile.target-build.v1",
        "state": "captured",
        "captured_at": datetime.now(timezone.utc).replace(
            microsecond=0).isoformat(),
        "source": {
            "deeptile_revision": revision,
            "source_dirty": dirty,
        },
        "toolchain": {"vitis_version": vitis_version(args.vitis_version)},
        "settings": parse_settings(args.setting),
        "architecture_defines": read_integer_defines(
            args.arch_config, ARCH_DEFINES),
        "abi": {
            "name": read_string_define(
                args.abi_header, "DEEPTILE_KERNEL_ABI_NAME"),
            "defines": read_integer_defines(args.abi_header, ABI_DEFINES),
        },
        "configuration_hashes": {
            "architecture_config_sha256": sha256_file(args.arch_config),
            "abi_header_sha256": sha256_file(args.abi_header),
            "kernel_config_sha256": sha256_file(args.kernel_config),
            "vivado_impl_config_sha256": sha256_file(args.vivado_config),
        },
        "inputs": sorted(inputs, key=lambda record: record["path"]),
    }


def finalize(args) -> dict:
    record = json.loads(args.capture.read_text(encoding="utf-8"))
    if (record.get("schema") not in ("deeptile.target-build.v1",
                                     "tilecnn.target-build.v1") or
            record.get("state") != "captured"):
        raise ValueError("input is not a captured DeepTile target build record")
    if not args.xclbin.is_file():
        raise FileNotFoundError(f"XCLBIN not found: {args.xclbin}")
    record["state"] = "complete"
    record["completed_at"] = datetime.now(timezone.utc).replace(
        microsecond=0).isoformat()
    record["artifact"] = {
        "xclbin": args.xclbin.name,
        "sha256": sha256_file(args.xclbin),
    }
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.add_argument("--repo-root", type=Path, required=True)
    capture_parser.add_argument("--arch-config", type=Path, required=True)
    capture_parser.add_argument("--abi-header", type=Path, required=True)
    capture_parser.add_argument("--kernel-config", type=Path, required=True)
    capture_parser.add_argument("--vivado-config", type=Path, required=True)
    capture_parser.add_argument("--input", type=Path, action="append", default=[])
    capture_parser.add_argument("--setting", action="append", default=[])
    capture_parser.add_argument("--vitis-version")

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--capture", type=Path, required=True)
    finalize_parser.add_argument("--xclbin", type=Path, required=True)
    finalize_parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "capture":
            result = capture(args)
        else:
            result = finalize(args)
        atomic_write(args.output, result)
    except (OSError, ValueError, json.JSONDecodeError,
            subprocess.CalledProcessError) as exc:
        print(f"target build record: {exc}")
        return 2
    print(f"Target build record {result['state']}: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
