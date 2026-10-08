# DeepTile compiler

The DeepTile compiler consumes a `tilecnn.graph.v1` model package and a hardware
configuration, lowers the graph into kernel launches, selects legal tiles and
schedules, and emits the transfer and compute descriptors consumed by the FPGA
kernel.

## Layout

```text
include/deeptile/compiler/   Graph types, loader, and graph compiler API
include/deeptile/tile/       Layer, schedule, descriptor, and cost-model API
src/                        Compiler implementations
tools/                      Graph, schedule, traffic, and cost inspectors
```

The graph compiler calls the tile scheduler internally. Scheduling is a
compiler subsystem, not an independently deployed framework component.

## Native build

From the repository root:

```bash
cmake -S scripts -B build/tools -DCMAKE_BUILD_TYPE=Debug
cmake --build build/tools -j
ctest --test-dir build/tools --output-on-failure
```

Important targets are `deeptile_compiler`, `graph_plan`, `graph_schedule`,
`schedule_inspect`, `ifm_traffic`, and `compiler_tests`.

The graph handoff contract is documented in
[`docs/reference/graph-handoff-spec.md`](../../docs/reference/graph-handoff-spec.md).
