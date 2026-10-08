#!/usr/bin/env python3
"""Generate a DeepTile TargetPackage manifest for one completed XCLBIN."""

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


# Build records under either namespace; tilecnn.* is the former name of the
# same format (docs/reference/naming.md).
BUILD_RECORD_SCHEMAS = ("deeptile.target-build.v1", "tilecnn.target-build.v1")

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
        match = re.search(rf"^#define\s+{name}\s+(\d+)\s*(?://.*)?$", text, re.M)
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


def architecture_key(values: dict[str, int]) -> str:
    fields = (
        ("bitwidth", values["BITWIDTH"]),
        ("gap_scale_frac_bits", values["GAP_SCALE_FRAC_BITS"]),
        ("ic_tile_size", values["IC_TILE_SIZE"]),
        ("oc_tile_size", values["OC_TILE_SIZE"]),
        ("p_ic", values["P_IC"]),
        ("p_spatial", values["P_SPATIAL"]),
        ("max_tile_height", values["MAX_TILE_H"]),
        ("max_tile_width", values["MAX_TILE_W"]),
        ("max_kernel_size", values["K_MAX"]),
        ("ifm_cache_depth", values["IFM_CACHE_DEPTH"]),
        ("weight_buffer_depth", values["WEIGHT_BUF_DEPTH"]),
        ("bias_buffer_depth", values["BIAS_BUF_DEPTH"]),
    )
    return ";".join(f"{name}={value}" for name, value in fields)


def repository_state(repo_root: Path) -> tuple[str, bool]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        text=True,
    ).strip())
    return revision, dirty


def inspect_xclbin(
    xclbin: Path,
    uuid_override: str | None,
    kernel_name: str,
    compute_unit: str,
    argument_count: int,
    platform_name: str,
) -> str:
    """Validate the embedded kernel metadata and return the XCLBIN UUID."""
    if uuid_override:
        return uuid_override
    output = subprocess.check_output(
        ["xclbinutil", "--info", "--input", str(xclbin)],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    match = re.search(r"UUID \(xclbin\):\s+([0-9a-fA-F-]+)", output)
    if match is None:
        raise ValueError("xclbinutil did not report an XCLBIN UUID")
    uuid = match.group(1).lower()

    signature = re.search(
        r"^\s*Signature:\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", output, re.M
    )
    if signature is None or signature.group(1) != kernel_name:
        raise ValueError(f"XCLBIN does not contain kernel signature {kernel_name}")
    arguments = [item for item in signature.group(2).split(",") if item.strip()]
    if len(arguments) != argument_count:
        raise ValueError(
            f"XCLBIN kernel has {len(arguments)} arguments; ABI declares "
            f"{argument_count}"
        )
    if re.search(rf"^\s*Instance:\s+{re.escape(compute_unit)}\s*$", output,
                 re.M) is None:
        raise ValueError(f"XCLBIN does not contain compute unit {compute_unit}")
    if platform_name not in output:
        raise ValueError(f"XCLBIN metadata does not identify platform {platform_name}")
    return uuid


def vitis_version(override: str | None) -> str:
    if override:
        return override
    output = subprocess.check_output(
        ["v++", "--version"], text=True, stderr=subprocess.STDOUT
    )
    match = re.search(r"v(\d{4}\.\d+)", output)
    return match.group(1) if match else "unknown"


def verified_build_record(path: Path | None, xclbin: Path) -> dict | None:
    if path is None or not path.is_file():
        print(
            "WARNING: no completed build record; target provenance is "
            "unverified"
        )
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        artifact = record["artifact"]
        if (record.get("schema") not in BUILD_RECORD_SCHEMAS or
                record.get("state") != "complete"):
            raise ValueError("record is not complete")
        if (artifact.get("xclbin") != xclbin.name or
                artifact.get("sha256") != sha256_file(xclbin)):
            raise ValueError("record does not belong to this XCLBIN")
        for field in (
                "source", "toolchain", "settings", "architecture_defines",
                "abi", "configuration_hashes"):
            if not isinstance(record.get(field), dict):
                raise ValueError(f"record is missing {field}")
        return record
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            f"WARNING: build record is unusable ({exc}); target provenance "
            "is unverified"
        )
        return None


def attach_boot_image(
    manifest: Path, boot_image: Path, declared_path: str | None = None
) -> dict:
    """Seal a packaged boot image into an existing manifest.

    The boot image is produced by packaging, which runs after the manifest is
    generated, so its digest cannot be known at generation time. Sealing it
    afterwards keeps one manifest describing both the XCLBIN and the image
    that actually programs the fabric.
    """
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    if not boot_image.is_file():
        raise FileNotFoundError(f"boot image not found: {boot_image}")

    image_path = Path(declared_path) if declared_path else Path(boot_image.name)
    if image_path.is_absolute() or ".." in image_path.parts:
        raise ValueError("boot image path must stay within the target package")

    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["boot"] = {
        "image": image_path.as_posix(),
        "sha256": sha256_file(boot_image),
    }
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # Generate-mode arguments are optional at parse time because
    # --attach-boot needs none of them; generate() validates its own inputs.
    parser.add_argument("--attach-boot", type=Path,
                        help="Seal this boot image into an existing manifest "
                             "and exit; requires --manifest")
    parser.add_argument("--manifest", type=Path,
                        help="Manifest to update with --attach-boot")
    parser.add_argument("--boot-path",
                        help="Package-relative path recorded for --attach-boot; "
                             "defaults to the boot-image filename")
    parser.add_argument("--xclbin", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--arch-config", type=Path)
    parser.add_argument("--abi-header", type=Path)
    parser.add_argument("--kernel-config", type=Path)
    parser.add_argument("--vivado-config", type=Path)
    parser.add_argument("--build-record", type=Path)
    parser.add_argument("--board", default="zcu104")
    parser.add_argument("--platform")
    parser.add_argument("--vitis-mode", choices=("hw", "hw_emu", "sw_emu"))
    parser.add_argument("--target-version", default="dev")
    parser.add_argument("--kernel-name", default="tile_conv")
    parser.add_argument("--compute-unit", default="tile_conv_1")
    parser.add_argument("--clock-hz", type=int)
    parser.add_argument("--psum-buffering",
                        choices=("ping-pong", "single"))
    parser.add_argument("--instrumentation",
                        choices=("none", "profile"), default="none")
    parser.add_argument("--uuid", help="Override UUID extraction for testing")
    parser.add_argument("--vitis-version",
                        help="Override Vitis version extraction for testing")
    return parser


def generate(args) -> dict:
    required = {
        "--output": args.output,
        "--repo-root": args.repo_root,
        "--arch-config": args.arch_config,
        "--abi-header": args.abi_header,
        "--kernel-config": args.kernel_config,
        "--vivado-config": args.vivado_config,
        "--platform": args.platform,
        "--vitis-mode": args.vitis_mode,
        "--clock-hz": args.clock_hz,
        "--psum-buffering": args.psum_buffering,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        raise ValueError("missing required arguments: " + ", ".join(missing))
    repo_root = args.repo_root.resolve()
    xclbin = args.xclbin.resolve()
    output = args.output.resolve()
    if not xclbin.is_file():
        raise FileNotFoundError(f"XCLBIN not found: {xclbin}")
    if output.parent != xclbin.parent:
        raise ValueError("target.json must be generated beside its XCLBIN")
    if not args.target_version.strip() or "@" in args.target_version:
        raise ValueError("target version must be non-empty and cannot contain @")
    if args.clock_hz <= 0:
        raise ValueError("kernel clock must be positive")

    record = verified_build_record(args.build_record, xclbin)
    if record:
        arch = record["architecture_defines"]
        abi = record["abi"]["defines"]
        abi_name = record["abi"]["name"]
        source = record["source"]
        revision = source.get("deeptile_revision", source.get("tilecnn_revision"))
        dirty = record["source"]["source_dirty"]
        settings = record["settings"]
        board = settings["board"]
        platform = settings["platform"]
        build_vitis_mode = settings["vitis_mode"]
        kernel_name = settings["kernel_name"]
        compute_unit = settings["compute_unit"]
        clock_hz = int(settings["clock_hz"])
        psum_buffering = settings["psum_buffering"]
        instrumentation = settings["instrumentation"]
        build_vitis_version = record["toolchain"]["vitis_version"]
        config_hashes = record["configuration_hashes"]
        provenance = {
            "verification": "verified",
            "deeptile_revision": revision,
            "source_dirty": dirty,
            "source_revision_scope": "fpga-build-record",
            "vitis_version": build_vitis_version,
            "architecture_config_sha256":
                config_hashes["architecture_config_sha256"],
            "abi_header_sha256": config_hashes["abi_header_sha256"],
            "kernel_config_sha256":
                config_hashes["kernel_config_sha256"],
            "vivado_impl_config_sha256":
                config_hashes["vivado_impl_config_sha256"],
            "build_record_sha256": sha256_file(args.build_record),
        }
    else:
        arch = read_integer_defines(args.arch_config, ARCH_DEFINES)
        abi = read_integer_defines(args.abi_header, ABI_DEFINES)
        abi_name = read_string_define(
            args.abi_header, "DEEPTILE_KERNEL_ABI_NAME")
        revision, dirty = repository_state(repo_root)
        board = args.board
        platform = args.platform
        build_vitis_mode = args.vitis_mode
        kernel_name = args.kernel_name
        compute_unit = args.compute_unit
        clock_hz = args.clock_hz
        psum_buffering = args.psum_buffering
        instrumentation = args.instrumentation
        build_vitis_version = vitis_version(args.vitis_version)
        provenance = {
            "verification": "unverified",
            "deeptile_revision": revision,
            "source_dirty": dirty,
            "source_revision_scope": "manifest-generation-worktree",
            "vitis_version": build_vitis_version,
            "architecture_config_sha256": sha256_file(args.arch_config),
            "abi_header_sha256": sha256_file(args.abi_header),
            "kernel_config_sha256": sha256_file(args.kernel_config),
            "vivado_impl_config_sha256": sha256_file(args.vivado_config),
        }

    psum_suffix = "pp" if psum_buffering == "ping-pong" else "single"
    target_name = f"{board}-p{arch['P_SPATIAL']}-{psum_suffix}"
    if instrumentation == "profile":
        target_name += "-profiled"
    target_id = f"{target_name}@{args.target_version}"

    key = architecture_key(arch)
    fingerprint = "sha256:" + hashlib.sha256(key.encode("utf-8")).hexdigest()
    platform_name = Path(platform).stem
    artifact_uuid = inspect_xclbin(
        xclbin,
        args.uuid,
        kernel_name,
        compute_unit,
        abi["DEEPTILE_KERNEL_ARGUMENT_COUNT"],
        platform_name,
    )
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    provenance.update({
        "generated_by": "scripts/hw/generate_target_manifest.py",
        "generated_at": generated_at,
    })

    return {
        "schema": "deeptile.target.v1",
        "target": {
            "name": target_name,
            "version": args.target_version,
            "id": target_id,
        },
        "platform": {
            "board": board,
            "vitis_platform": platform_name,
        },
        "artifact": {
            "vitis_mode": build_vitis_mode,
            "xclbin": xclbin.name,
            "sha256": sha256_file(xclbin),
            "uuid": artifact_uuid,
        },
        "kernel": {
            "name": kernel_name,
            "compute_unit": compute_unit,
            "clock_hz": clock_hz,
            "abi": {
                "name": abi_name,
                "version": abi["DEEPTILE_KERNEL_ABI_VERSION"],
                "argument_count": abi["DEEPTILE_KERNEL_ARGUMENT_COUNT"],
                "compute_descriptor_bytes":
                    abi["DEEPTILE_COMPUTE_DESCRIPTOR_BYTES"],
                "transfer_descriptor_bytes":
                    abi["DEEPTILE_TRANSFER_DESCRIPTOR_BYTES"],
            },
        },
        "architecture": {
            "parallelism": {
                "ic_tile_size": arch["IC_TILE_SIZE"],
                "oc_tile_size": arch["OC_TILE_SIZE"],
                "p_ic": arch["P_IC"],
                "p_spatial": arch["P_SPATIAL"],
            },
            "limits": {
                "max_tile_height": arch["MAX_TILE_H"],
                "max_tile_width": arch["MAX_TILE_W"],
                "max_kernel_size": arch["K_MAX"],
                "ifm_cache_depth": arch["IFM_CACHE_DEPTH"],
                "weight_buffer_depth": arch["WEIGHT_BUF_DEPTH"],
                "bias_buffer_depth": arch["BIAS_BUF_DEPTH"],
            },
            "numeric": {
                "bitwidth": arch["BITWIDTH"],
                "gap_scale_frac_bits": arch["GAP_SCALE_FRAC_BITS"],
            },
            "fingerprint": fingerprint,
        },
        "capabilities": {
            "model_schemas": ["deeptile.graph.v1", "tilecnn.graph.v1"],
            "residual_add": True,
            "max_pool": True,
            "global_average_pool": True,
            "fused_depthwise": True,
        },
        "implementation": {
            "psum_buffering": psum_buffering,
            "instrumentation": instrumentation,
        },
        "provenance": provenance,
    }


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.attach_boot is not None:
            if args.manifest is None:
                raise ValueError("--attach-boot requires --manifest")
            manifest = attach_boot_image(
                args.manifest, args.attach_boot, args.boot_path
            )
            args.output = args.manifest
        else:
            if args.xclbin is None:
                raise ValueError("--xclbin is required")
            manifest = generate(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary_output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_output.replace(args.output)
    except (KeyError, OSError, ValueError,
            subprocess.CalledProcessError) as exc:
        print(f"target manifest: {exc}")
        return 2
    if args.attach_boot is not None:
        print(f"TargetPackage boot image sealed: {args.output} "
              f"({manifest['boot']['image']} "
              f"sha256:{manifest['boot']['sha256'][:12]})")
    else:
        print(f"TargetPackage ready: {args.output} "
              f"({manifest['target']['id']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
