# Quick start

This page starts with a hardware-free validation of the framework, then takes
you through the optional model-export, HLS, and ZCU104 stages. It assumes the
tools for the stage you choose are installed as described in
[Installation](installation.md). To rebuild every artifact behind the paper
evaluation, see
[Reproduce the paper results](../guides/reproduce-results.md).

## 1. Clone the framework

```bash
git clone --recursive https://github.com/mokweri/DeepTile.git
cd DeepTile
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

## 2. Build and test the native framework

```bash
make -C scripts tools
ctest --test-dir build/tools --output-on-failure
```

Exercise the compiler and runtime planner with a checked-in ModelPackage:

```bash
./build/tools/runtime_dry_run data/resnet50_int8_tilecnn/graph.json
./build/tools/graph_schedule data/resnet50_int8_tilecnn/graph.json
```

At this point the native framework is installed and usable without AMD tools
or an FPGA board. Continue only for the parts of the full flow you need.

## 3. Fetch and export a model

With the FixQuant environment from [Installation](installation.md) active and
`FIXQUANT_PYTHON` exported, fetch and export a released model:

```bash
make -C scripts model-fetch \
    ZOO_MODEL=resnet50/imagenet1k/int8-tqt@v1.0.0
make -C scripts model-export \
    ZOO_MODEL=resnet50/imagenet1k/int8-tqt@v1.0.0
```

The verified package is published under
`build/models/resnet50/imagenet1k/int8-tqt/v1.0.0/`. Fetching is explicit;
export never downloads a checkpoint implicitly.

## 4. Run the HLS correctness gate

Initialize the Vitis 2024.2 environment, then run C simulation against the
exported integer reference:

```bash
make -C scripts hls-csim DEEPTILE_CASE=best_case_3x3_s1
```

Synthesis and co-simulation options are listed in the
[command reference](../reference/commands.md).

## 5. Build the standard hardware target

The repository default is the P4 ping-pong configuration:

```bash
make -C scripts hw-all VITIS_MODE=hw TARGET_VERSION=v2.0.0
```

The complete target is generated under `build/hw/zcu104-p4-pp_v2.0.0/`,
including the matching `deeptile` host, `deeptile.xclbin`, `target.json`, build
provenance, and SD-card package. The FPGA build of P4 takes about half an hour; see
[Build a hardware target](../guides/build-a-target.md) for other configurations.

## 6. Deploy and run

If this is the board's first DeepTile build, first create its bootable SD card by
following [Prepare the ZCU104](board-setup.md). Configure the `zcu104` SSH
alias, then deploy the complete target:

```bash
TARGET_DIR="$PWD/build/hw/zcu104-p4-pp_v2.0.0" \
    ./scripts/hw/deploy_zcu104.sh
ssh zcu104 'sudo reboot'
```

!!! warning "Reboot after a full deployment"
    `BOOT.BIN`, not the XCLBIN alone, establishes the active fabric
    configuration. The runtime checks that the deployed boot image crossed a
    reboot boundary before execution.

On the board, a packaged single-input reference run has the following form:

```bash
./deeptile target.json data/resnet50_int8_tilecnn/graph.json \
    --runs 3 --output results/resnet50-single
```

!!! tip "Running MobileNetV2 and EfficientNet-Lite0"
    Set `DEEPTILE_FUSE_DW=1` for depthwise-separable networks to enable fused
    producer → depthwise execution. Without it the results are still correct,
    but slower.

Next, [Run a model](../guides/run-a-model.md) covers deployment and run options
in detail, and [Evaluate on a dataset](../guides/evaluate-on-a-dataset.md)
measures accuracy on ImageNet.
