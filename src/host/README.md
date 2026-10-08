# DeepTile host

The DeepTile host combines a compiled deployment plan with the canonical tensor
payloads from a FixQuant model package. It packs tensors, plans device memory,
executes graph launches, profiles the FPGA, and compares outputs with the
integer reference.

## Layout

```text
main.cpp       Board application entry point
runtime/       Buffer planning, tensor packing, and graph orchestration
reference/     Bit-exact software reference execution
backends/      XRT board execution and HLS C-simulation adapter
tools/         Host-side inspection utilities
```

The host depends on `deeptile_compiler`; the optional XRT backend depends on the
host. The HLS adapter is compiled only by the Vitis HLS testbench flow.

## Build

Build native host utilities and tests:

```bash
make -C scripts tools
make -C scripts runtime_tests
```

Cross-compile the board executable:

```bash
make -C scripts hw-host VITIS_MODE=hw
```

The board executable remains `deeptile`. Hardware build and deployment details
are documented in [`scripts/README.md`](../../scripts/README.md).
Its standard first argument is the packaged `target.json`; a direct XCLBIN path
is retained only for legacy development runs.

## Run outputs

Pass `--output DIR` to retain a minimal, versioned execution record. By default,
single-image mode writes `run.json` and `launch-profile.csv`; dataset mode
writes `run.json` and `predictions.csv`. The record contains model and target
identities, the embedded DeepTile revision and dirty state, relevant execution
settings, the principal correctness or accuracy and timing results, and paths
to its companion artifacts.

It also carries one digest per consumed artifact — `identity.model_sha256`,
`identity.xclbin_sha256`, `identity.boot_sha256`, and in dataset mode
`execution.dataset_sha256` — so a published record can be checked against the
published packages instead of trusted. Absent fields mean the run had no such
artifact. Per-artifact checksums and build provenance remain in the
ModelPackage and TargetPackage manifests that own them.

```bash
./deeptile target.json graph.json --runs 3 --output results/single
./deeptile target.json graph.json --dataset data/imagenet \
    --output results/imagenet
```

An output directory may contain other files, but an existing `run.json` is not
overwritten.
