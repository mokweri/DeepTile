# Testing and validation

DeepTile changes are validated in proportion to the boundary they touch. Native
tests run in a minute; HLS simulation takes minutes; a hardware build takes
half an hour to several hours. Run the cheapest check that covers your change first.

## The reference oracle

Release-grade model packages carry integer reference outputs exported by
FixQuant under `refs/`. **These references are the pass/fail oracle for exact
integer validation.** HLS C simulation and packaged single-input board runs
compare the accelerator output against them, and any mismatch is a failure.
Graph-only developer packages may omit them; dataset runs report aggregate
classification accuracy rather than comparing each image with `refs/`.

DeepTile also contains a software reference implementation, used for debugging.
It deliberately mirrors the kernel's integer arithmetic, so it cannot
independently confirm the kernel: a shared mistake would satisfy both. It is
therefore never used as a pass/fail oracle. Instead, the
`reference_equivalence_tests` gate requires the software reference to reproduce
every checked-in exported reference exactly, with and without depthwise fusion.

## Native tests

```bash
make -C scripts tools
ctest --test-dir build/tools --output-on-failure
```

The native suite covers the compiler, host runtime, preprocessing, dataset
handling, the execution pipeline, metrics, target packages, and the reference
equivalence gate. It needs no AMD tools and no board. Individual suites include
`compiler_tests`, `host_tests`, `preprocess_tests`, `dataset_tests`,
`pipeline_tests`, `metrics_tests`, and `reference_equivalence_tests`.

## HLS simulation

```bash
# C simulation against the exported reference
make -C scripts hls-csim DEEPTILE_CASE=best_case_3x3_s1

# Add synthesis and co-simulation
make -C scripts hls-cosim DEEPTILE_CASE=best_case_3x3_s1 DEEPTILE_TRACE_LEVEL=none
```

Passing C simulation does not guarantee that co-simulation passes. Stream
deadlocks, FIFO-depth problems, and some memory hazards appear only in
co-simulation or on hardware, so co-simulate any change to streaming or
buffering behavior. Vitis HLS runs one project at a time; do not start a second
simulation against the same project.

Test cases live under `data/`. The single-layer cases in
`data/phase1_testcases/` and `data/p2_testcases/` are quick; whole networks are
slow.

## What to run for a change

| Change | Minimum validation |
|---|---|
| Compiler or host C++ | Native build and the relevant tests |
| Software reference or quantization arithmetic | `reference_equivalence_tests` across all checked-in packages |
| Graph import | Compiler tests against checked-in model packages |
| Compiler-to-host interface | Compiler tests, host tests, and `runtime_dry_run` |
| Descriptor ABI or shared configuration | ABI tests, a host build, and HLS C simulation |
| Kernel behavior | Focused testbench and HLS C simulation; co-simulation when timing or streaming matters |
| XRT integration or packaging | AArch64 host build (`make -C scripts hw-host`) and a target-package check |
| A new hardware target | Build, package, and one board run through `target.json` |

A packaged single-input board run is the final exact-reference acceptance test:
the host compares its output with the exported reference, exercising the whole
model-to-board path for that input.
