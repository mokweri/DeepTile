---
hide:
  - toc
---

# DeepTile

**Reusable INT8 CNN inference with fixed FPGA hardware and target-aware
software.**

DeepTile is an open-source hardware–software framework that runs quantized
CNNs on an FPGA accelerator built once and then reused. A target-aware compiler
maps each supported network onto the implemented target through descriptors,
and a runtime drives the accelerator directly, so changing networks needs a new
compilation, not a new FPGA build. Model export, compilation, deployment,
validation and measurement are connected by versioned, checksummed artifacts.

![DeepTile: trained networks are quantized and verified by FixQuant into ModelPackages, compiled by one target-aware compiler, and run on a single ZCU104 FPGA target, with no FPGA rebuild when switching networks.](assets/figures/deeptile-hero-light.svg#only-light)
![DeepTile: trained networks are quantized and verified by FixQuant into ModelPackages, compiled by one target-aware compiler, and run on a single ZCU104 FPGA target, with no FPGA rebuild when switching networks.](assets/figures/deeptile-hero-dark.svg#only-dark)

<div class="grid cards" markdown>

-   :material-chip:{ .lg .middle } **One target, many networks**

    ---

    The accelerator, ABI, and host/compiler configuration are fixed within a
    completed target. Changing networks changes the compiled schedule,
    parameters, and tensor organization, not the implemented FPGA design.

-   :material-check-decagram:{ .lg .middle } **One reference oracle**

    ---

    Release-grade ModelPackages carry integer reference outputs exported by
    FixQuant. HLS and packaged single-input board validation compare against
    them, and any mismatch is fatal.

-   :material-package-variant-closed:{ .lg .middle } **Explicit artifact contracts**

    ---

    ModelPackage, TargetPackage, and RunRecord bind every result to the model,
    target, dataset, and source state that produced it.

    [:octicons-arrow-right-24: How DeepTile works](concepts/how-it-works.md)

-   :material-chart-bar:{ .lg .middle } **Measured across targets**

    ---

    Five CNNs run on five implemented targets with one bitstream per target,
    showing how the value of added width and buffering depends on the
    network.

    [:octicons-arrow-right-24: Research](research.md)

</div>

## Where to start

| If you want to | Read |
|---|---|
| Set up the tools | [Installation](getting-started/installation.md) |
| Prepare a ZCU104 for its first deployment | [Prepare the ZCU104](getting-started/board-setup.md) |
| Build the tools and run a model | [Quick start](getting-started/quick-start.md) |
| Understand how the framework fits together | [How DeepTile works](concepts/how-it-works.md) |
| Run your own network | [Bring your own model](guides/bring-your-own-model.md) |
| Measure accuracy on ImageNet | [Evaluate on a dataset](guides/evaluate-on-a-dataset.md) |
| Build a different hardware configuration | [Build a hardware target](guides/build-a-target.md) |
| Reproduce the paper evaluation | [Reproduce the paper results](guides/reproduce-results.md) |

## Current scope

DeepTile is a research artifact rather than a prebuilt accelerator service. The
validated implementation currently targets:

| Property | Current support |
|---|---|
| FPGA platform | AMD/Xilinx ZCU104 (`XCZU7EV`) |
| Toolchain | Vitis/Vivado 2024.2 |
| Kernel clock | 200 MHz |
| Arithmetic | Signed INT8 activations and weights with wider accumulation |
| Execution | Batch-one, descriptor-driven tiled inference |
| Validated networks | ResNet-50, ResNet-18, MobileNetV2, EfficientNet-Lite0, and custom `vgg16_tilecnn` |
| Dense kernels | Square, up to 7x7; stride 1 or 2 |
| Other operations | Fused 3x3 and 5x5 depthwise convolution, residual add, max pool, global average pool, ReLU and ReLU6, and output requantization |
| Implemented targets | P2 and P4 ping-pong; P2, P4 and P8 single-buffer |

## Results

All five networks run on every target from the same bitstreams, evaluated on
3,923 ImageNet-mini validation images with DeepTile v2.0.1. Accuracy is
identical on every target. The times below are accelerator kernel time per
image in milliseconds, the sum of layer latencies under direct launch control.
P2-S, P4-S and P8-S differ in spatial width alone; each S/PP pair differs in
partial-sum buffering alone. MobileNetV2 and EfficientNet-Lite0 reach
application throughputs of up to 172 and 163 images per second.

| Network | Top-1 | P2-S | P2-PP | P4-S | P4-PP | P8-S |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 78.61% | 49.44 | 46.37 | 31.38 | 28.46 | 23.43 |
| ResNet-18 | 69.46% | 20.90 | 20.41 | 12.40 | 11.93 | 9.00 |
| MobileNetV2 | 69.95% | 6.32 | 5.27 | 5.02 | 4.43 | 4.49 |
| EfficientNet-Lite0 | 74.00% | 7.24 | 6.15 | 5.57 | 4.91 | 4.91 |
| VGG-class (`vgg16_tilecnn`) | 71.86% | 177.71 | 174.27 | 100.29 | 96.86 | 63.42 |

The custom VGG-class model is not stock VGG-16. Predictions are
byte-identical across all five targets.
[Reproduce the paper results](guides/reproduce-results.md) regenerates these
measurements on your own board.

## Width and workload

At equal partial-sum buffering, each target doubles the compute of the one
before. The grey track is that added compute; the coloured bar is the kernel
speedup each network actually achieves over P2-S.

![Kernel speedup over P2-S against added compute at single buffering. P4-S (2x compute): ResNet-50 1.58x, MobileNetV2 1.26x. P8-S (4x compute): ResNet-50 2.11x, MobileNetV2 1.41x.](assets/figures/deeptile-scaling-light.svg#only-light)
![Kernel speedup over P2-S against added compute at single buffering. P4-S (2x compute): ResNet-50 1.58x, MobileNetV2 1.26x. P8-S (4x compute): ResNet-50 2.11x, MobileNetV2 1.41x.](assets/figures/deeptile-scaling-dark.svg#only-dark)

Dense networks convert more of the added width into speed than
depthwise-separable ones, which depend more on ping-pong partial sums. A
reusable target should therefore be chosen for its intended workload mix
rather than by theoretical peak alone.
