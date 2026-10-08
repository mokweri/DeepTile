# Build a hardware target

A target is one implemented configuration of the accelerator: an FPGA binary,
the boot image that programs it, the matching host executable, and a
TargetPackage manifest describing all three. This guide builds one and explains
the configuration choices.

## Choose a configuration

| Choice | How it is set | Values |
|---|---|---|
| Spatial parallelism $P_{\mathrm{spatial}}$ | `P_SPATIAL` in `src/common/arch_config.h` | A power of two; 2, 4, and 8 have been built and evaluated |
| Partial-sum buffering | `PSUM_BUFFERING=` at build time | `ping-pong` (default) or `single` |
| Version | `TARGET_VERSION=` at build time | Any label; the evaluated targets are `v2.0.0` |
| Instrumentation | `PROFILE=1` at build time | Off by default |

$P_{\mathrm{spatial}}$ sets the compute width: the target performs
$P_{\mathrm{spatial}} \times 16 \times 16$ multiply–accumulates per cycle, for a
theoretical peak of $102.4 \times P_{\mathrm{spatial}}$ GOPS at 200 MHz. Wider
targets use more DSP slices and logic, and take longer to place and route.
Ping-pong partial-sum buffering lets accumulation of one tile overlap the
output processing of the previous one, at the cost of extra block RAM.

More compute does not help every network equally. At equal buffering, each
doubling of $P_{\mathrm{spatial}}$ speeds up the dense networks (ResNet-50,
ResNet-18, `vgg16_tilecnn`) by 1.34–1.77×, but MobileNetV2 and
EfficientNet-Lite0 by only 1.12–1.30×. Those two depend more on ping-pong
partial sums: single buffering makes them 13–20% slower at P2 and P4, against
2–10% for the dense networks. Choose width and buffering for the networks you
intend to run.

!!! info "Why `P_SPATIAL` is a source constant"
    `arch_config.h` is the single source of truth for the architecture. The
    hardware build reads `P_SPATIAL` from it to name the target, so a build
    directory can never disagree with the configuration it was built from. A
    make-variable override would allow them to drift apart.

## Build

The build needs the ZCU104 platform and AMD's matching ZynqMP common image. If
this is the first build on a workstation or the board does not yet have a
bootable DeepTile SD card, follow
[Prepare the ZCU104](../getting-started/board-setup.md).

The repository default, P4 with ping-pong buffering, builds from an unmodified
tree:

```bash
make -C scripts hw-all VITIS_MODE=hw TARGET_VERSION=v2.0.0
```

For another width, edit the header, build, and restore it:

```bash
sed -i 's/^#define P_SPATIAL 4$/#define P_SPATIAL 8/' src/common/arch_config.h
make -C scripts hw-all VITIS_MODE=hw TARGET_VERSION=v2.0.0 PSUM_BUFFERING=single
sed -i 's/^#define P_SPATIAL 8$/#define P_SPATIAL 4/' src/common/arch_config.h
```

!!! warning "Restore `arch_config.h` after every build"
    Restore the header on every exit path, including a failed build. A
    forgotten edit silently changes the next target you build.

`hw-all` runs three steps you can also invoke separately:

| Target | Step | Time |
|---|---|---|
| `hw-host` | Cross-compile the AArch64 host executable | Minutes |
| `hw-fpga` | Compile, link, and implement the FPGA design; generate `target.json` | About 30 min for P2 or P4; several hours for P8, whose routing is congested |
| `hw-package` | Assemble the package, seal the boot-image digest, then generate the flashable image | Minutes |

Vitis runs one build at a time. Do not start a second hardware build while one
is running.

Hardware packaging seals `BOOT.BIN` into both manifests before generating
`sd_card_hw/sd_card.img`. The flashable image and loose deployment directory
therefore contain the same manifest and boot identity.

## What you get

The build directory is named from the board, the width, the buffering, and the
version:

```text
build/hw/zcu104-p4-pp_v2.0.0/
|-- deeptile               # AArch64 host executable, compiled for this width
|-- deeptile.xclbin        # FPGA binary
|-- target.json           # TargetPackage manifest: identity, ABI, capabilities, checksums
|-- target-build.json     # build provenance, including the architecture-config hash
`-- sd_card_hw/sd_card/   # deployable SD-card package, including BOOT.BIN
```

The build captures the source revision, configuration hashes, and toolchain
before FPGA compilation and binds that record to the result, so `target.json`
reports its provenance as `verified`. Check a finished target without a board:

```bash
./build/tools/target_inspect build/hw/zcu104-p4-pp_v2.0.0/target.json
```

Native tools are compiled for the selected architecture too. Rebuild them with
the matching header before inspecting another width; the default P4 inspector
intentionally rejects P2 and P8 packages. Their matching board hosts validate
those packages at execution time.

!!! note "A target is complete only once packaged"
    `BOOT.BIN` is created by `hw-package`. A build directory holding only
    `deeptile.xclbin` and `target.json` describes hardware that cannot yet be
    made to run. Always deploy a target with its own `deeptile` executable; each
    one is compiled for its target's width.

## Validate before a long build

The HLS C simulation checks the kernel against an exported reference in a
couple of minutes, far faster than a hardware build:

```bash
make -C scripts hls-csim DEEPTILE_CASE=best_case_3x3_s1
```

Co-simulation (`hls-cosim`) also exposes stream deadlocks and FIFO-depth
problems that C simulation cannot, at the cost of several minutes per case.

## Emulation

To exercise the host and kernel without a board, build for software or
hardware emulation and run:

```bash
make -C scripts hw-run VITIS_MODE=hw_emu
```

## Next steps

- [Run a model](run-a-model.md) to deploy and run the new target.
- [Prepare the ZCU104](../getting-started/board-setup.md) for the one-time SD
  card setup.
- [Port to another board](port-to-another-board.md) to build for other
  hardware.
