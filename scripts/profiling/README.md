# Profiling and validation tools

These tools preserve measurement scope, artifact identity, and raw evidence.
They do not change scheduling policy or require model-specific hardware.
Use fresh output directories: evidence capture refuses to overwrite results.
All Python tools provide `--help`.

| Tool | Runs on | Purpose |
|---|---|---|
| `capture_hls_evidence.py` | Workstation | Preserve passing RTL reports and decode process/FIFO/loop observations |
| `report_final_checkpoint.tcl` | Vivado | Read-only timing, utilization, and bus-skew reports from the final DCP |
| `capture_implementation.py` | Workstation | Bind routed reports and actual kernel clock to an XCLBIN |
| `run_board_suite.py` | ZCU104 | Repeated exact-output and batch-one latency runs, optionally with power or tracing |
| `summarize_board.py` | Either | Validate raw RunRecords/CSV and summarize host timing and whole-board energy |
| `compare_runs.py` | Either | Compare matched kernel/compiler ablations with explicit controls |
| `read_am_counters.py` | ZCU104, root | Read and validate accelerator-monitor counters on an instrumented target |
| `summarize_device.py` | Either | Cross-check monitor counts against XRT summaries and the implemented clock |
| `isolate_graph.py` | Workstation | Extract operators using intermediate tensors bound to exported outputs |
| `run_compiler_ablation.py` | ZCU104 | Compare two hosts on one fabric, binding a fixture before/after each run |

Native `build/tools/descriptor_dump graph.json` emits size-tagged descriptor
bytes for same-machine compiler comparisons, not a deployment-plan ABI.

## C-model and bounded-FIFO regression checks

Build the optional runner with
`cmake --build build/tools --target deeptile_csim -j4`. Run it through
`python3 scripts/hls/run_csim_checks.py --executable build/tools/deeptile_csim --output
build/validation/cmodel --repeats 2 GRAPH...` to retain exact-output logs.
C-model streams are unbounded: C simulation does not prove RTL progress.

Independent scalar-oracle generators live in `tests/integration/fixtures/`:

- `generate_loader_boundaries.py`: alignment, stride, halo, and tails.
- `generate_epoch_probe.py`: transfer-depth probes with optional spatial size.
- `generate_chunk_suite.py`: cached partial-chain count/start boundaries.
- `generate_activation_cases.py`: ReLU6 clamps on dense, fused depthwise and
  post-add paths, and 5x5 dense and depthwise kernels.

Run an isolated RTL progress test from the repository root:

```bash
python3 tests/integration/fixtures/generate_epoch_probe.py build/validation/epochs
DEEPTILE_GRAPH="$PWD/build/validation/epochs/graph.json" \
DEEPTILE_HLS_COMPONENT=epoch_progress DEEPTILE_HLS_REPEATS=2 \
DEEPTILE_FORCE_EPOCHS=256 DEEPTILE_DATAFLOW_PROFILE=1 \
    make -C scripts hls-cosim DEEPTILE_TRACE_LEVEL=none
python3 scripts/profiling/capture_hls_evidence.py build/hls/epoch_progress \
    --output build/validation/epoch-rtl-evidence \
    --source-note 'Record the exact synthesized source/configuration identity here' \
    --log build/hls/epoch_progress.log
```

`DEEPTILE_HLS_EPOCH_SWEEP=1,2,31,32,33,256,1` repeats each requested schedule
`DEEPTILE_HLS_REPEATS` times. AXI wrapper depths must cover the largest schedule.
Capture results before reuse; never run two HLS processes on the same component.

Loader selection is no longer configurable. The accepted parity-proven bound
and its required row-dependent fallback are the canonical implementation.
`DEEPTILE_DISABLE_K1_CACHE`, `DEEPTILE_DISABLE_CIG_CHAIN`, and
`DEEPTILE_FORCE_EPOCHS` remain compiler diagnostics, not release defaults.

## Repeated board measurements

Copy the runner to `tools/profiling/` and optionally
`scripts/hw/deeptile_power_capture.py` to `tools/hw/` on the board. Deploy a
complete TargetPackage and reboot first. From the board's boot
directory, with the four existing `models/<model>/.../v1.0.0` packages:

```bash
python3 tools/profiling/run_board_suite.py --host ./deeptile --target target.json \
    --output results/release-latency --runs 500 --warmup 10 --repeats 3
```

Defaults select MobileNetV2, ResNet-50, ResNet-18, and custom VGG16 with fused
depthwise lowering. `--graphs` selects isolated fixtures with unique parent
directory names. The runner pins affinity (default CPU 2), records governor and
frequency, validates BOOT/XCLBIN hashes and the reboot boundary, and strips
inherited `DEEPTILE_*` overrides. It does not set CPU frequency: configure and
verify that separately before matched measurements.

For whole-board INA226 power, validate sensors with the existing power tools,
run `sudo -v`, and add `--power-script tools/hw/deeptile_power_capture.py`. Root is
needed for sensor configuration. The timing window comes from the successful
RunRecord, excluding warmup and preparation, not the full process lifetime.
Summarize a copied suite with:

```bash
python3 scripts/profiling/summarize_board.py results/release-latency \
    --output results/release-summary.json
```

## Optional stall tracing

`PROFILE=1` inserts AXI/stall monitors into a separate `-profiled` target. Use
uninstrumented targets for primary latency/energy. `--capture-profile` gives
each process its own trace directory;
`--counter-reader tools/profiling/read_am_counters.py`
adds read-only AM capture on a matching instrumented target after `sudo -v`.

The helper validates target, mapping, execution count, and register values. It
reads the latch first and retains high counter bits. An exit-zero but empty XRT
debug report is not evidence. Monitor counts include warmup, unlike timed runs.

RTL stalls overlap and are not additive cycle partitions. Read stalls include
descriptor/input starvation, not only AXI service. Coarse board monitor stalls
are not detailed FIFO stalls. Host launch/wait is not device cycles or isolated
layer latency. Graph wall covers prepacked input upload through packed output
download, excluding preparation, preprocessing, and final unpacking/comparison.
Whole-board energy is not PL-only. Repeat means are not population confidence
intervals; instrumented results cannot establish uninstrumented performance.

## Routed implementation and tests

Run `vivado -mode batch -source scripts/profiling/report_final_checkpoint.tcl
-tclargs FINAL.dcp NEW_REPORT_DIR`, then `capture_implementation.py TARGET_DIR
--checkpoint-reports NEW_REPORT_DIR --output NEW_EVIDENCE_DIR`. This preserves
checkpoint identity, actual clock, resources, setup/hold/pulse timing, and
bus skew. HLS estimated frequency is not timing closure.

CTest includes the decoder/measurement and fixture-generator tests. They also
run without FPGA tools:

```bash
python3 -B -m unittest discover -s tests/profiling -v
python3 -B -m unittest discover -s tests/integration/fixtures -v
```
