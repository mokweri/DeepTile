# Research

DeepTile is the subject of a journal manuscript in preparation. This page
summarizes its evaluation; [Reproduce the paper results](guides/reproduce-results.md)
regenerates every measurement on your own board.

<div class="grid cards" markdown>

-   :material-file-document-outline:{ .lg .middle } **DeepTile framework paper**

    ---

    The integrated accelerator, compiler and runtime; software gains on an
    unchanged bitstream; and a controlled study of spatial width and
    partial-sum buffering across five networks and five targets, with
    same-board comparisons against an AMD DPU core and cgra4ml.

    [:octicons-arrow-right-24: Reproduce the results](guides/reproduce-results.md)

</div>

## Current evaluation

The evaluation runs ResNet-50, ResNet-18, MobileNetV2, EfficientNet-Lite0 and
custom `vgg16_tilecnn` on five targets (P2 and P4 ping-pong; P2, P4 and P8
single-buffer). All 25 network–target combinations pass the bundled reference
checks, and each network's predictions are identical on every target.

- **Software on an unchanged bitstream.** On P4-PP, direct launch control,
  pointwise–depthwise fusion and stem folding together reduce kernel time by
  1.73× for MobileNetV2 and 1.63× for EfficientNet-Lite0.
- **Spatial width.** At single buffering, the first width doubling gives
  1.58–1.77× kernel speedup for the three dense networks and 1.26–1.30× for
  the two depthwise-separable ones.
- **Partial-sum buffering.** At fixed width, ping-pong buffering reduces
  kernel time by 12–17% for the depthwise-separable networks and 2–9% for the
  dense ones.
- **Same board.** One AMD DPU core has a shorter execution time on every
  network, while P4-PP reaches a higher fraction of its arithmetic peak.
  cgra4ml runs three of the five networks, and Arm processing dominates its
  inference time.

The [profiling and stall-tracing tools](https://github.com/mokweri/DeepTile/tree/main/scripts/profiling)
measure per-layer latency on your own targets.

## Citation

The DeepTile framework manuscript is currently in preparation:

```bibtex
@unpublished{mogaka2026deeptile,
  author = {Mogaka, Obed M. and Forsberg, H{\aa}kan and Daneshtalab, Masoud},
  title  = {{DeepTile}: Reusable {INT8} {CNN} Inference with Fixed {FPGA}
            Hardware and Target-Aware Software},
  note   = {Manuscript in preparation},
  year   = {2026}
}
```

If your work also uses FixQuant to quantize and export models, please cite the
FixQuant paper as well:

```bibtex
@inproceedings{mogaka2025fixquant,
  author    = {Mogaka, Obed M. and Forsberg, H{\aa}kan and Daneshtalab, Masoud},
  title     = {Bridging Quantization and Deployment: A Fixed-Point Workflow for
               {FPGA} Accelerators},
  booktitle = {2025 IEEE 28th International Symposium on Design and Diagnostics
               of Electronic Circuits and Systems (DDECS)},
  pages     = {123--126},
  year      = {2025},
  publisher = {IEEE},
  doi       = {10.1109/DDECS63720.2025.11006791}
}
```

DeepTile is licensed under the Apache License 2.0. The
[FixQuant](https://github.com/mokweri/FixQuantTool) submodule is a separate
project and retains its own license.
