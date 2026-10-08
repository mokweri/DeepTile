# Reproduce the paper results

This guide rebuilds the evaluation behind the DeepTile framework paper from a
clean checkout: five models, five hardware targets, and the dataset evaluation
runs. The paper's measurements are summarized on the [Research](../research.md)
page; [What to compare](#what-to-compare) lists what should match exactly.

The evaluated targets are `v2.0.0` hardware driven by the current host. The
kernel sources have not changed since those targets were built, so building
the current tree as `v2.0.0` reproduces the evaluated design.

Set up the tools first, as described in
[Installation](../getting-started/installation.md), including the FixQuant
environment and `FIXQUANT_PYTHON`.

## 1. Build and test the native tools

```bash
make -C scripts tools
ctest --test-dir build/tools --output-on-failure
```

The suite includes the reference-equivalence gate, which requires the software
reference to reproduce every exported reference exactly.

## 2. Fetch and export the five models

```bash
for m in resnet50/imagenet1k/int8-tqt@v1.0.0 \
         resnet18/imagenet1k/int8-tqt@v1.0.0 \
         mobilenet_v2/imagenet1k/int8-tqt-cle@v1.0.0 \
         efficientnet_lite0/imagenet1k/int8-tqt-cle-relu6@v1.0.0 \
         vgg16_tilecnn/imagenet1k/int8-tqt@v1.0.0; do
    make -C scripts model-fetch  ZOO_MODEL=$m
    make -C scripts model-export ZOO_MODEL=$m
done
```

Export is deterministic: re-exporting a release produces a byte-identical
`manifest.json`.

## 3. Check the kernel in C simulation

```bash
make -C scripts hls-csim DEEPTILE_CASE=best_case_3x3_s1
```

## 4. Build the five targets

The evaluation uses five configurations. Three width points at single
buffering isolate spatial width, and the two ping-pong targets isolate
partial-sum buffering at P2 and P4.

| Target | `P_SPATIAL` | `PSUM_BUFFERING` | Boot image SHA-256 (prefix) |
|---|---|---|---|
| `zcu104-p2-pp@v2.0.0` | 2 | `ping-pong` | `a5d98aaeda9d…` |
| `zcu104-p4-pp@v2.0.0` | 4 | `ping-pong` | `db99e11b6571…` |
| `zcu104-p8-single@v2.0.0` | 8 | `single` | `62c4d40bcd5f…` |
| `zcu104-p2-single@v2.0.0` | 2 | `single` | `f71129de5b5d…` |
| `zcu104-p4-single@v2.0.0` | 4 | `single` | `959c397806f9…` |

Build each one in turn, restoring the architecture header afterwards:

```bash
sed -i 's/^#define P_SPATIAL 4$/#define P_SPATIAL <N>/' src/common/arch_config.h
make -C scripts hw-all VITIS_MODE=hw TARGET_VERSION=v2.0.0 PSUM_BUFFERING=<mode>
git checkout src/common/arch_config.h
```

Each target's `target-build.json` records the header's digest, which ties a
build to its exact configuration. P2 and P4 take about half an hour each; P8
takes several hours. Run the builds one at a time.

The boot-image digests above identify the evaluated builds in their run
records. A rebuilt target carries its own digests in its manifest and
RunRecords; the FPGA binary in particular embeds a fresh UUID and timestamps on
every build.

## 5. Prepare the dataset

The paper uses the 3,923-image ImageNet-mini evaluation set. Prepare and stage
it as in [Evaluate on a dataset](evaluate-on-a-dataset.md). The `val.txt` for
that set has SHA-256
`b7de7986ec2a4a507a7565ebedac0fc9d03b1bd2ef9140616aaa5d19f999d350`.

## 6. Run each target

For each target, deploy its full package, reboot, and confirm the boot image:

```bash
TARGET_DIR=$PWD/build/hw/<target> ./scripts/hw/deploy_zcu104.sh
ssh zcu104 'sudo reboot'
ssh zcu104 'sha256sum ~/boot/BOOT.BIN'      # must equal boot.sha256 in target.json
```

Copy the five models to the board as in [Run a model](run-a-model.md), then run
each with three preprocessing workers and a queue of six:

```bash
# ResNet-50, ResNet-18 and vgg16_tilecnn
./deeptile target.json models/resnet50/imagenet1k/int8-tqt/v1.0.0/graph.json \
    --dataset data/imagenet --topk 5 --workers 3 --queue 6 \
    --output results/<target>/resnet50

# MobileNetV2 and EfficientNet-Lite0
DEEPTILE_FUSE_DW=1 \
./deeptile target.json models/mobilenet_v2/imagenet1k/int8-tqt-cle/v1.0.0/graph.json \
    --dataset data/imagenet --topk 5 --workers 3 --queue 6 \
    --output results/<target>/mobilenet_v2
```

Before comparing with the paper, check that each record states
`launch_mode: direct` and `stem_space_to_depth: true` (except for
`vgg16_tilecnn`, whose stem is stride 1), and that MobileNetV2 and
EfficientNet-Lite0 report `fused_depthwise: true` with 36 and 34 launches.

Write records to the SD card, not `/tmp`, which the reboot before the next
target clears.

The paper reports timing, throughput, accuracy, and routed resources. Optional
power measurements are documented separately in
[Measure power and energy](power-and-energy.md).

## What to compare

- **Predictions** should match exactly: for each network, `predictions.csv` is
  byte-identical across all five targets and across worker counts.
- **Accuracy** follows from the predictions: top-1 is 78.61% for ResNet-50,
  69.46% for ResNet-18, 69.95% for MobileNetV2, 74.00% for EfficientNet-Lite0
  and 71.86% for `vgg16_tilecnn` on this image set.
- **Kernel time** varies slightly from run to run; compare the dataset mean
  with the record's tables. See
  [Execution and timing](../concepts/execution-and-timing.md).
- **Identities**: each `run.json` records the digests of the model manifest,
  FPGA binary, and boot image, so you can confirm which artifacts produced a
  result.
