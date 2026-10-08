# DeepTile scripts

This directory contains the developer-side build, HLS, hardware packaging, and
ZCU104 deployment workflows. Run the examples below from the repository root
unless a section says otherwise.

Reusable RTL stall capture, board tracing, routed-report capture, exact-output
suites, and measurement tools are documented in
[`profiling/README.md`](profiling/README.md). Independent regression generators
live in `tests/integration/fixtures/`. The optimized loader is the default;
there are no experimental loader-selection flags.

Hardware commands keep generated caches, logs, run scripts, and package outputs
inside their selected `build/hw/<target>/` directory. HLS invocation logs live
beside their component under `build/hls/`, not in this source directory.

## Prerequisites

- Native tools: CMake and a C++17 compiler.
- HLS flows: Xilinx Vitis HLS with its environment initialized.
- Hardware flows: Xilinx Vitis/XRT, the ZCU104 platform, and the configured
  AArch64 sysroot. The defaults are in `hw/Makefile`.
- Board deployment: `ssh`, `tar`, and an SSH host named `zcu104`.

Show the main commands with:

```bash
make -C scripts help
```

## Native tools and tests

Build all native graph/runtime tools:

```bash
make -C scripts tools
```

Validate a packaged target, including its XCLBIN and boot image, without
opening an FPGA:

```bash
./build/tools/target_inspect build/hw/zcu104-p4-pp_v2.0.0/target.json
```

Build only the graph-plan tool used to derive HLS trip counts:

```bash
make -C scripts graph_plan
```

Build and run the native runtime tests:

```bash
make -C scripts runtime_tests
```

Check every model package under `data/` against its exported reference:

```bash
make -C scripts reference_tests
```

The exporter-provided tensors in `data/<package>/refs/` are the single oracle
for output correctness, so this is the gate to run after touching the software
reference, the quantization arithmetic, or a fixture generator. C-simulation
and the board host compare against those references alone.

Dataset images are not tracked. Derive the evaluation root once from a local
ImageNet validation split before any dataset command
(see [Evaluate on a dataset](../docs/guides/evaluate-on-a-dataset.md)):

```bash
./scripts/data/prepare_imagenet_subset.py \
    --source ~/Documents/datasets/imagenet-mini/val --out data/imagenet
```

Preprocess dataset images without touching the accelerator:

```bash
./build/tools/dataset_preprocess data/imagenet --limit 100 --convention eval
```

The tool indexes an ImageNet-style directory, decodes and preprocesses each
image on worker threads, and reports both CPU cost per image and wall-clock
throughput. `--workers N` overrides the default of two workers; output is
byte-identical at any worker count. Image decoding
needs OpenCV; a build without it still compiles and tests the preprocessing
arithmetic, and the tool reports the missing decoder rather than failing
silently. `--convention export` selects the exporter's squash geometry instead
of the standard evaluation transform. See
[Evaluate on a dataset](../docs/guides/evaluate-on-a-dataset.md).

Native outputs are written under `build/tools/`. Remove them with:

```bash
make -C scripts clean-tools
```

## Model-package export

Install the pinned FixQuant submodule into the active Python environment, then
fetch and export a validated model-zoo release:

```bash
python -m pip install --no-deps -e ./FixQuant
make -C scripts model-fetch \
    ZOO_MODEL=resnet50/imagenet1k/int8-tqt@v1.0.0
make -C scripts model-export \
    ZOO_MODEL=resnet50/imagenet1k/int8-tqt@v1.0.0
```

`model-fetch` downloads only a missing checkpoint, verifies its release-record
size and SHA-256, and installs it atomically in FixQuant's ignored artifact
cache. Fetching remains explicit so export and evaluation never make an
unexpected network request. `model-export` exports into a temporary directory,
verifies the complete package inventory and SHA-256 checksums for the graph,
parameters, input, and reference, validates the graph with the DeepTile compiler,
and publishes it only after all checks succeed. Repeating the command validates
and reuses an existing package with the same release ID without overwriting its
contents. The ready package follows the model-zoo identity:

```text
build/models/resnet50/imagenet1k/int8-tqt/v1.0.0/
|-- manifest.json
|-- graph.json
|-- inputs/
|-- params/
`-- refs/
```

Generated packages under `build/models/` are disposable framework artifacts.
The packages under `data/` remain curated regression fixtures and are never
overwritten by this command.

## HLS validation

The HLS entry point is `hls/run_hls.tcl`. The top-level Makefile supplies its
environment variables and invokes Vitis HLS.

```bash
# C simulation: correctness gate
make -C scripts hls-csim DEEPTILE_CASE=best_case_3x3_s1

# C simulation followed by synthesis
make -C scripts hls-csynth DEEPTILE_CASE=best_case_3x3_s1

# Fresh simulation, synthesis, and co-simulation
make -C scripts hls-cosim \
    DEEPTILE_CASE=best_case_3x3_s1 \
    DEEPTILE_TRACE_LEVEL=none

# Re-run co-simulation using an existing synthesis result
make -C scripts hls-cosim-reuse \
    DEEPTILE_CASE=best_case_3x3_s1 \
    DEEPTILE_TRACE_LEVEL=port

# Export the synthesized design
make -C scripts hls-export
```

Supported trace levels are `none`, `port`, and `all`. HLS outputs are written
under `build/hls/deeptile_hls_proj/`.

The generic HLS mode interface is also available:

```bash
make -C scripts hls DEEPTILE_HLS_EXEC=<1..6> DEEPTILE_CASE=<case>
```

The mode mapping is documented at the top of `hls/run_hls.tcl`. An explicit
graph can be supplied with `DEEPTILE_GRAPH=/path/to/graph.json`.

## Hardware build and package

The hardware Makefile is `hw/Makefile`; the top-level targets delegate to it.

```bash
# Cross-compile the host application
make -C scripts hw-host VITIS_MODE=hw

# Cross-compile the dataset preprocessing tool
make -C scripts hw-dataset VITIS_MODE=hw

# Build the FPGA binary and its target manifest
make -C scripts hw-fpga VITIS_MODE=hw TARGET_VERSION=v2.0.0

# Generate the complete SD-card package
make -C scripts hw-package VITIS_MODE=hw TARGET_VERSION=v2.0.0

# Run all three steps
make -C scripts hw-all VITIS_MODE=hw TARGET_VERSION=v2.0.0
```

For the standard P4 ping-pong target at version `v2.0.0`, the main hardware
artifacts are:

```text
build/hw/zcu104-p4-pp_v2.0.0/deeptile
build/hw/zcu104-p4-pp_v2.0.0/dataset_preprocess
build/hw/zcu104-p4-pp_v2.0.0/deeptile.xclbin
build/hw/zcu104-p4-pp_v2.0.0/target-build.json
build/hw/zcu104-p4-pp_v2.0.0/target.json
build/hw/zcu104-p4-pp_v2.0.0/sd_card_hw/sd_card/
```

The package directory contains the boot files, host executable, xclbin,
target manifest, runtime scripts, XRT configuration, and `data/`. `VITIS_MODE`
selects the Vitis execution mode (`hw`, `hw_emu`, or `sw_emu`); the manifest's
`target.id` identifies the hardware configuration independently of its build
directory. The default directory is derived from the board, `P_SPATIAL`,
partial-sum buffering, profiling state, and `TARGET_VERSION`; `BUILD_DIR` can
still override it for an exceptional build.

`hw-fpga` captures source, configuration, toolchain, and architecture state
before compilation, then binds that record to the linked XCLBIN. `target.json`
therefore reports `provenance.verification` as `verified`. Running
`hw-manifest` on a compatible existing XCLBIN without its build record is
allowed, but the resulting target is marked `unverified` and the inspection
and host paths print a warning.

The host executable serves both single-image and dataset runs from one binary:

```bash
# Single image: exact reference check plus run.json and launch-profile.csv
./deeptile target.json data/resnet50_int8_tilecnn/graph.json --runs 3 \
    --output results/resnet50-single

# Full dataset: accuracy plus run.json and predictions.csv
./deeptile target.json data/resnet50_int8_tilecnn/graph.json \
    --dataset data/imagenet --topk 5 --output results/resnet50-imagenet
```

Dataset mode reports top-1 and top-5 accuracy alongside accelerator time,
preparation cost, and the fraction of wall time the accelerator was busy.
`--workers N` overrides the default of two worker threads;
predictions are identical at any worker count. Run `./deeptile --help` for the
full option list. `--output` refuses to replace an existing `run.json`, making
each output directory one self-contained execution receipt. The older `--csv`
and `--predictions` options remain available when no run record is needed.

For emulation, select `sw_emu` or `hw_emu`. For example:

```bash
make -C scripts hw-run VITIS_MODE=hw_emu
```

Clean a hardware target with:

```bash
make -C scripts hw-clean VITIS_MODE=hw
```

## Deploy to the ZCU104

`hw/deploy_zcu104.sh` runs on the development PC. It checks the deployment
files, connects through SSH, verifies the real boot partition, and streams them
through `tar`. Remote `rsync` is not required.

Configure the board alias in `~/.ssh/config`:

```sshconfig
Host zcu104
    HostName <current-board-IP>
    User petalinux
```

Test the connection:

```bash
ssh zcu104
```

Package the target and current host, then deploy the complete boot package:

```bash
D=build/hw/zcu104-p4-pp_v2.0.0
make -C scripts/hw package VITIS_MODE=hw TARGET_VERSION=v2.0.0 \
    BUILD_DIR=$PWD/$D -o $PWD/$D/deeptile.xclbin -o $PWD/$D/deeptile
TARGET_DIR=$PWD/build/hw/zcu104-p4-pp_v2.0.0 \
    ./scripts/hw/deploy_zcu104.sh
ssh zcu104 'sudo reboot'
```

When `build/hw/` contains exactly one target manifest, `TARGET_DIR` may be
omitted. Runtime deployment copies only `deeptile`, `deeptile.xclbin`, and
`target.json`; it does not duplicate or replace unchanged boot assets. If the
target contains `sd_card_hw/sd_card/`, the script automatically performs a full
boot-package deployment instead. `SD_CARD_DIR` selects an explicit full package.

The default board settings are:

```text
SSH alias:           zcu104
Boot mount:          /run/media/boot-mmcblk0p1
Expected device:     /dev/mmcblk0p1
Local target:        auto-discovered under build/hw/, or TARGET_DIR
```

The safety check refuses to deploy unless the remote destination is a writable
mount backed by `/dev/mmcblk0p1`. This prevents accidentally copying files into
the unrelated `/boot` directory on the root filesystem.

The existing remote `data/` and `models/` directories are deliberately
preserved. Runtime deployment touches only the matched host, XCLBIN, and target
manifest. Full deployment excludes local `data/` and transfers the remaining
package contents, including boot files, runtime scripts, and XRT configuration.

Settings may be overridden when deploying a different target or board:

```bash
BOARD=zcu104 \
TARGET_DIR=$PWD/build/hw/zcu104-p4-pp_v2.0.0 \
REMOTE_BOOT_DIR=/run/media/boot-mmcblk0p1 \
EXPECTED_BOOT_DEVICE=/dev/mmcblk0p1 \
./scripts/hw/deploy_zcu104.sh
```

These overrides cover boards that use the same PetaLinux boot-partition model.
Porting DeepTile to another platform may also require a platform file, sysroot,
kernel image, boot packaging, and a board-specific deployment check. The new
build should still emit the same TargetPackage contract with its matching host
executable.

Reboot after every full deployment so the copied `BOOT.BIN` programs the PL:

```bash
ssh zcu104 'sudo reboot'
```

The SSH connection closing during reboot is normal. Allow the board time to
boot, initialize Ethernet, obtain its DHCP address, and start SSH. A host-only
runtime update does not require a reboot when its declared `BOOT.BIN` is already
active. Changing the hardware target always requires the matching full package
and a reboot; copying a new XCLBIN cannot change the fabric on this platform.

Copy an exported ModelPackage to the preserved `models/` directory. Keeping it
separate avoids embedding one network into every hardware package:

```bash
MODEL_REL=resnet50/imagenet1k/int8-tqt/v1.0.0
tar czf - -C build/models "$MODEL_REL" |
    ssh zcu104 'mkdir -p ~/boot/models && cd ~/boot/models && tar xzf -'
```

Later target deployments preserve this directory.

## Run on the board

After deployment, connect to the board and enter the actual boot partition:

```bash
ssh zcu104
cd /run/media/boot-mmcblk0p1
```

The PetaLinux image also provides `~/boot` as a symbolic link to that mount.
Do not confuse it with `/boot`, which is a different directory on this image.

Run the generated application script when the required graph data is present:

```bash
bash run_app.sh
```

Alternatively, invoke the host directly:

```bash
export LD_LIBRARY_PATH=/mnt:/tmp:${LD_LIBRARY_PATH:-}
export XILINX_XRT=/usr
./deeptile ./target.json data/<case>/graph.json \
    --runs 5 --warmup 2 --output results/<run-name>

./deeptile ./target.json \
    models/resnet50/imagenet1k/int8-tqt/v1.0.0/graph.json \
    --dataset data/imagenet --topk 5 --output results/resnet50-imagenet
```

### Comparing configurations

There is no sweep wrapper. Comparing hardware configurations means packaging
each target, deploying its full SD-card set, rebooting, and running the same
commands against each — because on this fixed platform the fabric comes from
`BOOT.BIN`, and a script that redeploys only an XCLBIN measures whichever
fabric the board already carries.

```bash
D=build/hw/zcu104-p4-pp_v2.0.0
make -C scripts/hw package VITIS_MODE=hw TARGET_VERSION=v2.0.0 BUILD_DIR=$PWD/$D \
     -o $PWD/$D/deeptile.xclbin -o $PWD/$D/deeptile
TARGET_DIR=$D VITIS_MODE=hw ./scripts/hw/deploy_zcu104.sh
ssh zcu104 'sudo reboot'
ssh zcu104 'sha256sum ~/boot/BOOT.BIN'   # must equal the target's boot.sha256
```

The host verifies the boot image and rejects one installed during the current
boot session. `deploy_zcu104.sh` applies the same checks before a runtime
deployment. Discard the first run after each boot; it carries a large one-time
cost. Copy `--output` records off the board before the next reboot, since `/tmp`
is tmpfs.

See the [TargetPackage reference](../docs/reference/target-package.md) for
why the boot image is part of a target.

Fixtures are never copied by the deploy script; upload a missing suite with
`tar czf - -C data <suite> | ssh zcu104 'cd /run/media/boot-mmcblk0p1/data && tar xzf -'`
(the board has no `rsync`).

## Python utilities

- `compute_tripcounts.py` derives synthesis trip-count and AXI-depth macros
  from a graph using `build/tools/graph_plan`.
- `tools/gen_dw_testcase.py` generates depthwise test fixtures.
- `tools/gen_pwdw_testcase.py` generates fused pointwise-depthwise fixtures.
- `tools/extract_kernel_manifest.py` extracts and validates timing information
  from a Vitis HLS `csynth.xml` report.
- `tools/p0_metrics.py` and the `p0_roofline_*.py` scripts reproduce analysis
  metrics and plots; the plotting scripts require Matplotlib.

Run any Python utility with `--help` for its full arguments.
