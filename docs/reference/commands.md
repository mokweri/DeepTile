# Command reference

All build commands run from the repository root through `make -C scripts`.
`make -C scripts help` prints the list.

The host's `--measure-latency` option records batch-one input upload through
packed output download. With `--output DIR`, it emits `graph-profile.csv` and
the timing scope in `run.json`. This excludes preprocessing and final reference
comparison and is separate from host launch/wait intervals. It applies only to
single-input mode, not dataset evaluation.

See the repository's `scripts/profiling/README.md` for supported RTL stall,
board counter, repeated measurement, and routed-implementation tools.

## Make targets

### Native tools and tests

| Target | Effect |
|---|---|
| `tools` | Configure and build the native compiler, host libraries, tools, and tests into `build/tools/` |
| `runtime_tests` | Build and run the runtime tests |
| `reference_tests` | Check every package under `data/` against its exported reference |
| `graph_plan` | Build only the graph-plan tool |
| `clean-tools` | Remove `build/tools/` |

Run the full native test suite with `ctest --test-dir build/tools --output-on-failure`.

### Models

| Target | Effect |
|---|---|
| `model-fetch ZOO_MODEL=<id>` | Download and verify a released FixQuant checkpoint |
| `model-export ZOO_MODEL=<id>` | Export and validate a ModelPackage into `build/models/` |

A release ID has the form `model/dataset/profile@version`, for example
`resnet50/imagenet1k/int8-tqt@v1.0.0`. Set `FIXQUANT_PYTHON` to the interpreter
that has FixQuant installed.

### HLS

| Target | Effect |
|---|---|
| `hls-csim` | C simulation: the correctness gate |
| `hls-csynth` | C simulation, then synthesis |
| `hls-cosim` | Fresh simulation, synthesis, and co-simulation |
| `hls-cosim-reuse` | Co-simulation reusing an existing synthesis result |
| `hls-export` | Export the synthesized design |
| `hls DEEPTILE_HLS_EXEC=<1..6>` | Generic mode: 1 C simulation, 2 plus synthesis, 3 plus co-simulation, 4 export, 5 C simulation of all cases, 6 co-simulation only |

| Variable | Meaning |
|---|---|
| `DEEPTILE_CASE` | Test case under `data/` to simulate, for example `best_case_3x3_s1` |
| `DEEPTILE_GRAPH` | Path to a `graph.json`; overrides `DEEPTILE_CASE` |
| `DEEPTILE_TRACE_LEVEL` | Co-simulation waveform tracing: `none`, `port`, or `all` |
| `DEEPTILE_FUSE_DW` | `1` to lower eligible pointwise → depthwise pairs as fused |

HLS projects are written to `build/hls/deeptile_hls_proj/`. Vitis HLS runs one
project at a time.

### Hardware

| Target | Effect |
|---|---|
| `hw-host` | Cross-compile the AArch64 host executable |
| `hw-dataset` | Cross-compile the dataset preprocessing tool |
| `hw-fpga` | Build the FPGA binary and its `target.json` |
| `hw-manifest` | Regenerate `target.json` beside an existing FPGA binary |
| `hw-package` | Assemble the SD-card package and seal the boot-image digest |
| `hw-all` | `hw-host`, `hw-fpga`, and `hw-package` |
| `hw-run` | Run under emulation (`VITIS_MODE=sw_emu` or `hw_emu`) |
| `hw-clean` | Remove a hardware build |

| Variable | Default | Meaning |
|---|---|---|
| `VITIS_MODE` | `hw` | `hw`, `hw_emu`, or `sw_emu` |
| `TARGET_VERSION` | `dev` | Version label of the target |
| `PSUM_BUFFERING` | `ping-pong` | `ping-pong` or `single` |
| `BOARD` | `zcu104` | Board name used in the target's identity |
| `PROFILE` | `0` | `1` builds an instrumented target |
| `NJOBS` | `8` | Parallel implementation jobs |
| `FUSE_DW` | unset | `1` makes the packaged `run_app.sh` export `DEEPTILE_FUSE_DW=1` (`hw-package`) |
| `BUILD_DIR` | derived | Override the build directory |
| `PLATFORM` | ZCU104 2024.2 base platform | Absolute path to the target board's `.xpfm` file |
| `COMMON_SW_DIR` | local installation | AMD common-image directory containing `Image`, `rootfs.ext4`, and the installed sysroot |

Spatial parallelism is set by `P_SPATIAL` in `src/common/arch_config.h`, not by
a make variable; see [Build a hardware target](../guides/build-a-target.md).
The one-time common-image and SD-card procedure is documented in
[Prepare the ZCU104](../getting-started/board-setup.md).

### Documentation

| Target | Effect |
|---|---|
| `docs-venv` | Install the documentation toolchain into `build/docs-venv` |
| `docs-serve` | Build the C++ API reference and preview the site |
| `docs` | Build the static site into `build/docs/site` |
| `docs-clean` | Remove generated documentation |

## Host executable

```text
deeptile <target.json> <graph.json> [options]
```

Single-image mode runs the graph's packaged input and compares the output with
its exported reference. Adding `--dataset` selects dataset mode.

| Option | Mode | Meaning |
|---|---|---|
| `--runs N` | Single image | Measured runs (default 1) |
| `--warmup N` | Single image | Warm-up runs (default 1) |
| `--csv FILE` | Single image | Write a per-launch profile |
| `--no-compare` | Single image | Skip the reference comparison |
| `--measure-latency` | Single image | Time input upload through output download for each run |
| `--dataset ROOT` | Dataset | ImageNet-style dataset root |
| `--limit N` | Dataset | Evaluate at most N images |
| `--workers N` | Dataset | Preprocessing threads (default 2) |
| `--queue N` | Dataset | Prepared images buffered (default twice the workers) |
| `--topk N` | Dataset | Report top-k alongside top-1 (default 5) |
| `--convention NAME` | Dataset | `eval` (default) or `export` |
| `--short-side N` | Dataset | Resize target for `eval` (default 256) |
| `--predictions FILE` | Dataset | Write per-image predictions |
| `--allow-missing` | Dataset | Continue when `val.txt` names absent images |
| `--output DIR` | Both | Write `run.json` and the mode's CSV; never overwrites a record |
| `--kernel NAME` | Both | Kernel name (default `tile_conv`) |
| `--device N` | Both | Device index (default 0) |
| `--verbose` | Both | Per-launch or per-image detail |

Passing an `.xclbin` instead of `target.json` is accepted for old scripts, but
bypasses every identity and compatibility check.

### Environment variables

| Variable | Effect |
|---|---|
| `DEEPTILE_FUSE_DW=1` | Enable fused producer → depthwise lowering. Recommended for MobileNetV2 and EfficientNet-Lite0 |
| `DEEPTILE_STEM_S2D=0` | Disable the space-to-depth fold of stride-2 RGB stems (on by default) |
| `DEEPTILE_XRT_LAUNCH=run` | Start launches through XRT's scheduler instead of direct register control. Per-launch time then becomes an initiation interval |
| `DEEPTILE_FORCE_EPOCHS=N` | Force N weight epochs per layer instead of the cost-model choice. Diagnostic |
| `DEEPTILE_PACK_EPOCHS=1` | Disable cost-guided epoch selection. Diagnostic |
| `DEEPTILE_DISABLE_K1_CACHE=1` | Disable the resident pointwise input cache. Diagnostic |
| `DEEPTILE_DISABLE_CIG_CHAIN=1` | Disable chained channel reductions. Diagnostic |

The diagnostic variables change the emitted descriptors, so a run that uses
them records them under `execution.schedule_flags`. RunRecords also state the
launch mode and stem folding under `execution.launch_mode` and
`execution.stem_space_to_depth`.

## Inspection tools

Built into `build/tools/` by `make -C scripts tools`.

| Tool | Usage | Purpose |
|---|---|---|
| `target_inspect` | `target_inspect <target.json>` | Verify a target's FPGA binary and boot image without a board |
| `runtime_dry_run` | `runtime_dry_run <graph.json>` | Prepare the full runtime plan without a device |
| `graph_schedule` | `graph_schedule <graph.json>` | Show how each layer is lowered and scheduled |
| `graph_plan` | `graph_plan <graph.json>` | Print the launch plan used to derive HLS trip counts |
| `schedule_inspect` | `schedule_inspect <layer> [--detail] [--descriptors] [--cost-v2]` | Inspect one built-in test layer's schedule, descriptors, and modelled cost |
| `ifm_traffic` | `ifm_traffic <graph.json> <policy>` | Report scheduled input-feature-map traffic |
| `descriptor_dump` | `descriptor_dump <graph.json>` | Emit size-tagged descriptor bytes for same-machine compiler comparisons; not a deployment ABI |
| `dataset_preprocess` | `dataset_preprocess <root> [--limit N] [--convention NAME]` | Measure preprocessing cost without the accelerator |

## Deployment script

`scripts/hw/deploy_zcu104.sh` deploys a target to the board over SSH.

| Variable | Default | Meaning |
|---|---|---|
| `TARGET_DIR` | the single target under `build/hw/` | Target to deploy |
| `SD_CARD_DIR` | from the target | Explicit SD-card package for a full deployment |
| `BOARD` | `zcu104` | SSH alias of the board |
| `REMOTE_BOOT_DIR` | `/run/media/boot-mmcblk0p1` | Boot partition mount on the board |
| `EXPECTED_BOOT_DEVICE` | `/dev/mmcblk0p1` | Block device the mount must be backed by |
