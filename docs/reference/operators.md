# Supported operators

This page lists the layers and shapes the current accelerator supports. The
compiler checks a graph against these rules when it loads it, so an unsupported
layer is reported before anything runs on the board. The exact schema fields
are defined in the [graph handoff specification](graph-handoff-spec.md).

## Numeric format

| Property | Support |
|---|---|
| Activations and weights | Signed 8-bit fixed point |
| Biases | Signed 8-bit, widened to 32 bits inside DeepTile |
| Scaling | One power-of-two scale per tensor, given as a fractional bit count (`frac`) |
| Accumulation | Wider than 8 bits, then rescaled and saturated to signed 8-bit |
| Batch size | One |

Per-channel scales and zero points are not supported.

## Operators

| Operator | Support | How it runs |
|---|---|---|
| `conv2d`, dense | Square kernels up to 7×7; equal strides of 1 or 2; no dilation; symmetric padding | One accelerator launch |
| `conv2d`, depthwise | `groups` equal to input and output channels; weights `[C, 1, K, K]` | Fused with the preceding dense convolution when eligible (3×3 or 5×5), otherwise as a diagonal dense convolution |
| Residual add | The shortcut matches the convolution output's shape; projection shortcuts supported | Fused into the convolution that produces the sum |
| ReLU, ReLU6 | After a convolution, or after a residual add | Fused into the convolution |
| `maxpool2d` | 3×3 kernel, stride 2, padding 1 | Fused into the preceding convolution |
| `gap2d` (global average pooling) | After a convolution | Fused into the preceding convolution |
| `linear` | Any size | Runs as a 1×1 convolution over one spatial position |

Grouped convolutions other than depthwise (`1 < groups < C`), dilated
convolutions, non-square kernels, and unequal strides are not supported. A
max-pooling or global-average-pooling layer that cannot be fused with a
preceding convolution may be rejected. A post-operation the compiler does not
implement is an error rather than being ignored.

## Fused depthwise execution

Depthwise-separable networks such as MobileNetV2 and EfficientNet-Lite0 pair a
1×1 convolution with a depthwise convolution.
When the pair is eligible, both run in one launch and the intermediate feature
map stays on chip. Eligible means:

- a 3×3 or 5×5 depthwise layer with same padding and stride 1 or 2;
- fed by a convolution without a residual add; a feature map wider than 64
  pixels is split into column tiles that overlap by the kernel's halo, and a
  strided producer such as a stem is tiled to fit its input buffer;
- at most 2,048 channels for 3×3, or 1,296 for 5×5, the capacity of the
  stage's weight store.

Ineligible depthwise layers fall back to diagonal dense execution, which is
correct but slower.

Fusion is enabled at run time with `DEEPTILE_FUSE_DW=1`; see
[Run a model](../guides/run-a-model.md#running-depthwise-separable-networks).

## Tiling limits

Output tiles are at most 32 × 64 positions, and kernels at most 7 × 7. These are
limits of the hardware buffers, not of your model: the compiler chooses a legal
tile for every layer and splits large layers into more launches. An exporter
never needs to consider them.

## Channel limits

Bias storage bounds output width; descriptor fields and cache capacity limit
which input widths can use the optional resident-cache path.

| Property | Limit | Set by |
|---|---|---|
| Output channels per layer | 8192 | Bias cache depth |
| Input channels, for an eligible 1×1 layer using the resident input cache | 6112 with chaining; 2048 unchained | Partial-chain count/start fields |

Exceeding the second limit is not an error: the compiler runs that layer without
the input cache, which is slower but produces identical results. Cache capacity,
padding, output-channel count, and tile geometry are additional eligibility
checks. The compiler splits reductions above 255 channel groups into two legal
partial chains; the external descriptor ABI is unchanged.

## Validated networks

ResNet-50, ResNet-18, MobileNetV2, EfficientNet-Lite0, and custom
`vgg16_tilecnn` are validated end to end on every implemented target, including
exact comparison against their exported integer references.
The VGG-class variant uses supported pooling and classifier geometry; stock
VGG-16's 2×2 pooling and original classifier are not covered by this validation.
Other networks built only
from the operators above are expected to work; confirm a new network with the
checks in
[Bring your own model](../guides/bring-your-own-model.md#check-that-the-model-fits),
and treat a clean compile as necessary rather than sufficient — the exported
reference comparison is what establishes correctness.
