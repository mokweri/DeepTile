# Hardware architecture

The DeepTile accelerator is a single tiled INT8 datapath that the compiler
time-multiplexes across every layer of a network. This page explains what the
main hardware blocks do, how a layer is mapped onto them, and which parameters
you choose when you build a target.

## Datapath and memory

![DeepTile accelerator datapath: descriptor control, the input loader above the dense MAC array, the ping-pong weight buffer beside it, the accumulate stage with single or ping-pong partial sums, and the output chain of requantization, residual add, pooling, depthwise stage and write-back.](../assets/figures/paper/deeptile-datapath.svg){ .paper-figure }

| Block | Role |
|---|---|
| **Descriptor control** | Reads the compiler's transfer and compute descriptors and drives every other stage |
| **Input loader** | Fetches input tiles into two banked on-chip stores (UltraRAM), so the next tile loads while the current one computes. For eligible pointwise layers one store holds the input tile as a resident cache |
| **Weight buffer** | Two block-RAM sets filled in *epochs*, so weight transfer for the next epoch overlaps computation on the current one |
| **Dense MAC array** | $P_{\mathrm{spatial}}$ spatial lanes × 16 output lanes, each cell a 16-input dot product. One activation vector is broadcast to every output lane and one weight-bank read serves every spatial lane. Two signed 8-bit products for adjacent output lanes share each DSP48E2 slice |
| **Accumulate** | Holds 32-bit partial sums on chip while successive input-channel groups contribute to an output tile. Single or ping-pong storage is a build-time choice |
| **Output chain** | Adds bias, requantizes and saturates to INT8, applies ReLU or ReLU6, residual addition, max pooling or global average pooling, as each launch requests |
| **Depthwise stage** | A line-buffered 3×3 or 5×5 depthwise convolution fed by the producing convolution's output stream; it is passed through when a launch does not use it |
| **Write-back** | Streams finished output tiles to external memory |

Inputs, dense weights, outputs and descriptors each have their own memory
port; residuals, biases and depthwise weights have theirs. Linear (fully
connected) layers run on the same array as $1 \times 1$ convolutions over a
single spatial position.

The array performs

$$
N_{\mathrm{MAC/cycle}} = P_{\mathrm{spatial}} \times P_{IC} \times P_{OC}
$$

multiply–accumulates per cycle. $P_{IC}$ and $P_{OC}$ are both 16;
$P_{\mathrm{spatial}}$ is the main parameter that distinguishes one target from
another.

## Mapping a layer onto the accelerator

A convolution layer is usually too large to hold on chip, so the compiler cuts
it into bounded pieces and orders them:

![Mapping one layer: (a) the input split into a spatial tile with its halo and into input-channel groups, weights as output-channel by input-channel blocks grouped into epochs, and the output in output-channel groups; (b) the execution order of epochs, spatial tiles, output-channel groups and input-channel groups, with the three reuse paths.](../assets/figures/paper/deeptile-layer-mapping.svg){ .paper-figure }

- **Spatial tiles** are regions of output positions, plus the surrounding
  input samples (the *halo*) that the kernel window needs.
- **Input-channel groups (ICGs)** hold $P_{IC} = 16$ channels; unused lanes of
  an incomplete final group are zero.
- **Output-channel groups (OCGs)** hold $P_{OC} = 16$ channels.
- **Weight epochs** are sets of OCGs whose weights fit one weight-buffer set.

The schedule visits **epochs, then spatial tiles, then OCGs, then ICGs**. Each
OCG accumulates every ICG and kernel position of its tile before the output
chain emits it. The compiler chooses tile sizes and epochs for each layer; the
target fixes the channel-group width and the buffer capacities.

## Reuse inside the accelerator

Three mechanisms avoid moving the same data twice. The compiler decides, per
layer, which to use; the hardware provides the buffering for all three.

1. **Input retention across output-channel groups.** For eligible pointwise
   ($1 \times 1$) layers, the input tile loaded for the first OCG serves every
   later OCG instead of being re-read.
2. **Accumulation chaining across input-channel groups.** Partial sums stay on
   chip while all ICGs of an output tile are processed, so intermediate sums
   are never written out between groups.
3. **Weight prefetching.** While one weight-buffer set feeds the array, the
   other loads the next epoch.

## Depthwise-separable layers

A depthwise convolution applies one filter per channel, which would waste most
of a dense array. DeepTile handles depthwise layers in two ways:

- **Fused producer → depthwise.** When a dense convolution feeds a $3 \times 3$
  or $5 \times 5$ depthwise layer, both run in a single launch. The depthwise
  stage consumes the producer's output directly from a four-row line buffer, so
  the intermediate feature map never leaves the chip. A row holds 64 pixels;
  wider feature maps are tiled by column, adjacent tiles overlapping by the
  kernel's halo. Each incoming column is multiplied by all five weight columns,
  two weights sharing one DSP; a $3 \times 3$ kernel uses the newest three rows
  and weight columns.
- **Diagonal-dense fallback.** Other depthwise layers run on the dense array as
  a convolution with a diagonal weight matrix. This is slower but always
  correct.

!!! warning "Fusion is enabled at run time"
    Fused lowering is enabled by setting `DEEPTILE_FUSE_DW=1` when you run the
    host. Without it, MobileNetV2 and EfficientNet-Lite0 still produce correct
    results, but compile to 53 and 50 launches instead of 36 and 34 and run more
    slowly. See [Run a model](../guides/run-a-model.md).

## Target parameters

These are fixed when you build a target and recorded in its TargetPackage.

| Parameter | Where it is set | Built values |
|---|---|---|
| Spatial parallelism, $P_{\mathrm{spatial}}$ | `P_SPATIAL` in `src/common/arch_config.h` | 2, 4, 8 (a power of two) |
| Partial-sum buffering | `PSUM_BUFFERING` at build time | `ping-pong` or `single` |
| Channel parallelism, $P_{IC}$ and $P_{OC}$ | `arch_config.h` | 16 |
| Maximum output tile | `MAX_TILE_H` × `MAX_TILE_W` in `arch_config.h` | 32 × 64 |
| Maximum kernel size | `K_MAX` in `arch_config.h` | 7 |
| Fused depthwise kernel | `DW_KMAX` in `arch_config.h` | 3 or 5 |
| Kernel clock | Hardware build | 200 MHz |

The theoretical arithmetic peak of a target follows directly:

$$
\text{peak GOPS} = 2 \times P_{\mathrm{spatial}} \times 16 \times 16 \times f_{\mathrm{clk}}
$$

At 200 MHz this is $102.4 \times P_{\mathrm{spatial}}$ GOPS. [Build a hardware
target](../guides/build-a-target.md) explains how to choose and build a
configuration.

## Where to go next

- [Execution and timing](execution-and-timing.md) — how launches run and how to
  read kernel timings.
- [Supported operators](../reference/operators.md) — exactly which layer types
  and shapes the hardware accepts.
