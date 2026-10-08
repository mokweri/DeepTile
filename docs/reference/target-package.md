# TargetPackage specification

This specification defines `target.json`, the manifest that identifies a built
hardware target. Read it if you build targets or
[port DeepTile to another board](../guides/port-to-another-board.md); running a
provided target needs none of it.

## Purpose

A TargetPackage binds one deployable XCLBIN to the hardware properties that
the compiler and host must know before using it. The package prevents a build
directory name or XCLBIN filename from being mistaken for proof of
compatibility.

The initial on-disk package is deliberately small:

```text
<target-directory>/
|-- deeptile.xclbin
`-- target.json
```

An SD-card package also carries the matching `deeptile` host executable and
platform boot files.

**The boot image is part of the contract, not just deployment material.** The
ZCU104 base platform is fixed, not DFX: the programmable logic is configured at
boot from the bitstream inside `BOOT.BIN`, and loading an XCLBIN afterwards only
registers metadata with `zocl`. The XCLBIN therefore does not determine which
hardware runs. Replacing it without its boot image leaves the previous fabric
active while the XCLBIN checksum, the XCLBIN UUID reported by `zocl`, and every
output comparison all still agree — because every architecture variant computes
identical results. A manifest that addressed only the XCLBIN would certify a
target that is not running.

A hardware TargetPackage therefore declares its boot image:

```json
"boot": {
  "image": "BOOT.BIN",
  "sha256": "<64 hex digits>"
}
```

The group is optional. Emulation modes have no boot image, and targets packaged
before the field existed remain loadable; both are treated as having no
verifiable fabric. When the group is present it must be complete, since a
half-declared boot image implies a verification that would not happen.

Because packaging produces `BOOT.BIN` after the manifest is generated, the
digest is sealed afterwards by
`generate_target_manifest.py --attach-boot <BOOT.BIN> --manifest <target.json>`,
which the `package` rule runs for both manifest locations. The manifest inside
the SD-card package records `BOOT.BIN`; the build-directory manifest records
`sd_card_hw/sd_card/BOOT.BIN`, so `target_inspect` can verify the package from
either location without adding another 21 MB copy beside the root manifest.

## Identity and naming

Three names serve different purposes:

| Name | Example | Meaning |
|---|---|---|
| Vitis mode | `hw` | Tool execution mode: `hw`, `hw_emu`, or `sw_emu` |
| Build directory | `build/hw/zcu104-p4-pp_v2.0.0` | Local, disposable output location |
| Target ID | `zcu104-p4-pp@v2.0.0` | Versioned hardware-package identity |

The Make variable is therefore `VITIS_MODE`, not `TARGET`. The manifest
generator derives `target.name` from the board, spatial parallelism, partial-
sum buffering, and instrumentation. `TARGET_VERSION` supplies its version;
`target.id` is always `<name>@<version>`.

Changing an implementation property represented in the name produces a
different target. Rebuilding an unchanged target for a release produces a new
version. Hardware builds normally use
`build/<VITIS_MODE>/<target-name>_<version>/`; the folder remains a convenience
and is never a compatibility input.

## Manifest fields

`target.json` uses schema identifier `deeptile.target.v1` and contains these
groups. The host also accepts `tilecnn.target.v1`, the same format under its
former name, in which the provenance field `deeptile_revision` is named
`tilecnn_revision`; see [Naming](naming.md).

| Group | Contract |
|---|---|
| `target` | Derived name, supplied version, and canonical ID |
| `platform` | Board and Vitis platform identity |
| `boot` | Boot image that programs the fabric, and its digest (optional; hardware only) |
| `artifact` | Relative XCLBIN path, SHA-256, UUID, and Vitis mode |
| `kernel` | Kernel/CU names, clock, and versioned binary ABI |
| `architecture` | Compiler-visible parallelism, limits, numeric format, and fingerprint |
| `capabilities` | Supported graph schemas and fused operations |
| `implementation` | Partial-sum buffering and instrumentation variants |
| `provenance` | Verified/unverified status, DeepTile revision, build scope, Vitis version, config and build-record hashes, and generator |

The architecture fingerprint is SHA-256 over a canonical ordered key defined
by `architecture_key()` in
`src/compiler/include/deeptile/tile/hardware_spec.hpp`. The Python generator
uses the same field order. It covers only properties that affect compilation;
artifact identity is enforced separately by the XCLBIN checksum and UUID.

Routed resource utilization is intentionally absent from version 1. It is
useful reporting metadata, but it is not needed to compile or safely load a
target. It can be added as an optional report-derived section once extraction
is automated.

Version 1 is currently enforced by the C++ loader and its contract tests. No
placeholder `schemas/` directory is needed. Add a standalone JSON Schema under
`src/compiler/schemas/` when FixQuant or another external tool needs to validate
this file without using the compiler library.

## Generation

The FPGA build owns manifest generation because it has the XCLBIN and the
effective implementation settings:

```bash
make -C scripts hw-fpga \
    VITIS_MODE=hw \
    TARGET_VERSION=v2.0.0 \
    PSUM_BUFFERING=ping-pong
```

Before FPGA compilation, `hw-fpga` captures the source revision, dirty state,
toolchain, architecture and ABI values, build settings, input-file hashes, and
configuration hashes. A successful link finalizes `target-build.json` with the
XCLBIN checksum, and manifest generation consumes that record. To generate or
refresh a manifest beside an existing XCLBIN without rebuilding hardware:

```bash
make -C scripts hw-manifest \
    VITIS_MODE=hw \
    TARGET_VERSION=v2.0.0 \
    PSUM_BUFFERING=ping-pong
```

When the matching completed build record exists, the refreshed manifest retains
verified build-time provenance. A compatible existing XCLBIN without such a
record is still packageable and runnable, but the generator marks its
provenance `unverified`; it must not be presented as a reproducible release
artifact. In that case the current headers and Make settings must still be
known to describe the XCLBIN because compatibility validation cannot recover
internal architecture constants from the binary.

The capture/finalize tool is `scripts/hw/target_build_record.py`, and the
manifest generator is `scripts/hw/generate_target_manifest.py`. For a verified
build, architecture, ABI, source, settings, and toolchain values come from the
completed build record. Without one, the generator falls back to the current
headers and worktree and labels the result unverified. UUID and XCLBIN checksum
are always measured from the packaged artifact.
The generator also reads the embedded XCLBIN metadata and refuses a package
whose kernel name, compute-unit name, argument count, or platform disagrees
with the requested contract.

## Loading and compatibility

The compiler loader under `src/compiler/src/target_loader.cpp` validates:

- schema and required field types;
- canonical target ID;
- a safe package-relative artifact path;
- a complete and safe boot group when one is declared;
- manifest architecture fingerprint;
- graph-schema support;
- the kernel ABI; and
- build-provenance classification; and
- agreement with the architecture compiled into the current compiler.

The host verifier under `src/host/runtime/target_artifact.cpp` resolves the
XCLBIN relative to `target.json` and checks its SHA-256 before XRT sees it. When
the manifest declares a boot image, it verifies that too. On the board this is
the boot partition's `BOOT.BIN`; a mismatch is fatal and reports that the full
package must be deployed and the board rebooted.

Full deployment also records the current Linux boot ID in
`.deeptile-deploy-boot-id`. A matching file checksum with the same boot ID means
the image has merely been copied and the previous fabric is still active, so
the host rejects execution until a reboot changes the boot ID.

`deploy_zcu104.sh` applies the checksum and boot-ID rules before a runtime
deployment. It refuses to ship an XCLBIN when the matching boot image is absent
or still pending activation, rather than producing a board that silently
executes the previous fabric.
`build/tools/target_inspect <target.json>` performs both checks without opening
an FPGA and prints the resolved identity, provenance classification, and
artifact. Missing build provenance is a warning, not a compatibility error.
The supported board invocation is:

```bash
./deeptile target.json data/<model-package>/graph.json [options]
```

A direct XCLBIN argument remains accepted for old developer scripts, but it
bypasses target identity and compatibility checks and is not the framework
path.

Version 1 still compiles architecture constants into the host/compiler. The
manifest therefore detects a mismatched package; it does not yet let one host
binary schedule arbitrary P2, P4, and P8 targets. Each TargetPackage therefore
carries the host executable built for that target. A single multi-architecture
host is outside the current framework scope.

## Kernel ABI evolution

`src/common/kernel_abi.h` is the source of truth for the externally visible
kernel ABI name, version, argument count, and descriptor sizes. Descriptor
layout assertions remain in the common/compiler ABI tests.

Use these rules when the kernel interface changes:

1. A change to argument count/order/type, descriptor byte layout, or descriptor
   interpretation requires incrementing `DEEPTILE_KERNEL_ABI_VERSION`.
2. Rebuild the host and XCLBIN, then regenerate the target manifest. The
   generator and loader read the new ABI constant automatically.
3. Add or update ABI assertions and run native tests, an AArch64 host build,
   and HLS C-simulation.
4. Keep an old loader branch only when backward compatibility is intentional;
   otherwise an old target must fail before kernel launch.

A normal ABI version bump does not require editing generator or loader logic.
Those components change only if the manifest representation or compatibility
policy itself changes.

## Validation gates

Changes to this contract require:

```bash
make -C scripts tools
ctest --test-dir build/tools --output-on-failure
make -C scripts hw-host VITIS_MODE=hw
make -C scripts hls-csim DEEPTILE_CASE=best_case_3x3_s1
```

For a new hardware build, also generate its manifest, package it, and complete
one board run through `target.json`. `tests/compiler/target_package_tests.cpp`
locks hashing, ABI and architecture rejection, package-path containment,
relative artifact resolution, checksum rejection, and boot-image verification
including the absent, mismatched, missing-file, and malformed cases.

A hardware target is not complete until it has been packaged. A build directory
holding only `target.json` and `deeptile.xclbin` describes hardware that cannot
be made to run, because the boot image that programs it does not exist yet.
