# Repository layout

DeepTile's source tree follows the framework's ownership boundaries: each stage
of the pipeline has one owner, and a change belongs where its behavior is
decided. This page is the map for anyone extending DeepTile.

## Top level

| Path | Contents |
|---|---|
| `src/common/` | The kernel ABI, packed descriptor and data types, and architecture configuration shared by hardware and software |
| `src/kernel/` | The HLS accelerator |
| `src/compiler/` | Graph import, lowering, tiling, scheduling, cost models, and descriptor generation |
| `src/host/` | Runtime planning, tensor packing, XRT execution, software reference, and dataset evaluation |
| `tests/` | Compiler, host, kernel, and cross-component tests |
| `scripts/` | Supported build, HLS, hardware, packaging, and deployment commands |
| `data/` | Checked-in model packages, HLS test cases, and dataset listings |
| `FixQuant/` | The FixQuant quantization and export tool, as a Git submodule |
| `docs/` | This documentation |
| `build/` | Generated models, tools, HLS projects, and hardware targets; ignored by Git |

## Components

**`src/common/` — the hardware/software contract.** Definitions that the
compiler, host, and kernel must interpret identically: `kernel_abi.h` declares
the ABI version and descriptor sizes, `common_defs.h` defines the packed
descriptors, and `arch_config.h` holds the architecture parameters. Nothing
belongs here unless more than one side of the hardware/software boundary reads
it.

**`src/kernel/` — the accelerator.** Top-level dataflow, descriptor
consumption, data movement, dense and depthwise computation, accumulation and
quantization, fused post-operations, and write-back. The kernel executes the
compiler's decisions and contains no layer-shape logic.

**`src/compiler/` — the mapping compiler.** Parsing and validating graphs and
TargetPackages, lowering operators to launches, choosing legal tiles and
schedules, cost estimation, and emitting transfer and compute descriptors. Its
public interface is under `src/compiler/include/deeptile/`.

**`src/host/` — deployment and execution.** Converting compiler output into a
runtime memory plan, loading and packing tensors, allocating buffers,
verifying target artifacts, running the graph through XRT, reconstructing
outputs, the software reference, and dataset evaluation.

The host executable links the compiler, so compilation happens in the host
process at start-up. Library dependencies run one way:

```text
deeptile executable -> deeptile_xrt -> deeptile_host -> deeptile_compiler
```

## Where a change belongs

| Change | Location |
|---|---|
| A graph field, operator import, or lowering rule | `src/compiler/` |
| Tile legality, traversal order, schedule policy, or cost model | `src/compiler/` |
| Tensor loading, packing, memory allocation, or orchestration | `src/host/runtime/` |
| Device interaction through XRT | `src/host/backends/` |
| Software reference behavior | `src/host/reference/` |
| Dataset loading, image preprocessing, or accuracy metrics | `src/host/dataset/` |
| A descriptor or packed type read by both hardware and software | `src/common/` |
| Datapath, on-chip buffering, streams, or kernel ports | `src/kernel/` |
| A correctness scenario spanning components | `tests/integration/` |
| A build or deployment command | `scripts/` |

## Rules that keep the design reusable

- **The kernel holds no model or layer policy.** It only executes descriptors.
- **Scheduling belongs to the compiler**, never to the host backend or the
  kernel.
- **Packing and execution preparation belong to the host**, not the compiler.
- **Shared ABI definitions have one source of truth**, with tests across the
  boundary. Changing a descriptor field, packed type, or architecture limit is a
  cross-boundary change: it needs ABI tests, HLS validation, a host rebuild, and
  a decision about existing FPGA binaries. Increment
  `DEEPTILE_KERNEL_ABI_VERSION` in `src/common/kernel_abi.h` whenever the kernel
  interface changes, so old targets fail cleanly before launch.
- **Source never depends on generated files** under `build/`.

## Next

See [Testing and validation](testing.md) for the checks each kind of change
needs.
