# Graph handoff specification

This specification defines the ModelPackage format: the files and metadata a
quantizer exports so that DeepTile can compile and run a network. Read it if you
are writing an exporter other than FixQuant. FixQuant users do not need it;
see [Bring your own model](../guides/bring-your-own-model.md).

DeepTile imports the graph, packs tensors into accelerator layouts, generates
schedules and descriptors, allocates buffers, and runs the kernel. The format is
graph-level and deliberately simple: signed 8-bit tensors in separate binary
files, explicit fixed-point metadata, and canonical software layouts. This page
describes schema version `deeptile.graph.v1`.

Packages exported before DeepTile was renamed carry `tilecnn.graph.v1`, the
same format under its former identifier, which readers still accept; see
[Naming](naming.md).

## Ownership Boundary

The quantization/export tool must provide:

- graph topology
- tensor shapes
- tensor file paths
- canonical tensor layouts
- quantization fractional bits
- signed int8 activation, weight, and bias files
- optional file-backed validation inputs and references

DeepTile is responsible for:

- converting activations from `CHW` into the packed accelerator layout
- converting convolution weights from `OIHW` into the accelerator's weight-bank layout
- converting linear weights from `OI` into a `1x1` convolution-compatible layout
- choosing legal tile shapes for each lowered hardware launch
- widening int8 biases to int32 internally
- applying DeepTile's hardware bias pre-shift internally
- deriving `shift_out`, `shift_bias`, `gap_mul`, and residual alignment shifts
- generating weight-transfer descriptors and compute descriptors
- allocating and reusing device buffers
- lowering graph nodes into one or more `tile_conv` launches

The quantization tool must not pre-pack tensors for DeepTile or encode DeepTile
channel-packing factors unless a future schema version explicitly adds a packed
export mode. Hardware packing and tiling are DeepTile compiler decisions.

## Recommended Directory Layout

Each exported model should be a directory with one graph JSON file and separate
binary tensor files.

```text
resnet18_int8_tilecnn/
  manifest.json
  graph.json

  inputs/
    input_0.int8.bin

  params/
    conv1.weight.int8.bin
    conv1.bias.int8.bin
    layer1.0.conv1.weight.int8.bin
    layer1.0.conv1.bias.int8.bin
    layer1.0.conv2.weight.int8.bin
    layer1.0.conv2.bias.int8.bin
    layer2.0.downsample.0.weight.int8.bin
    layer2.0.downsample.0.bias.int8.bin
    ...
    fc.weight.int8.bin
    fc.bias.int8.bin

  refs/
    logits.int8.bin
    optional/
      conv1_out.int8.bin
      layer1.0_out.int8.bin
      gap_out.int8.bin
```

Use separate files per tensor. This makes debugging, partial regeneration, and
per-layer comparison much easier than a single monolithic binary archive.

## Model Package Manifest

Released-model exports include `manifest.json` with schema
`deeptile.model-package.v1`. This package-level record identifies the model-zoo
release, FixQuant version and Git revision, checkpoint and configuration
checksums, reference-image preprocessing, and checksums for `graph.json` and
every file-backed parameter, validation input, and reference. New exports
declare SHA-256 coverage of all graph artifacts explicitly.

The preprocessing entry describes generation of the packaged single-input
reference. It is not a declaration of the ImageNet dataset-evaluation transform;
the DeepTile host selects and records that convention separately.

`manifest.json` describes how the package was produced. `graph.json` remains
the compiler input and graph/tensor contract. Consumers may ignore the package
manifest when reading legacy fixtures, but release-oriented workflows should
validate it before accepting a newly generated package.

The DeepTile export wrapper requires complete inventory coverage before
publishing a generated package. The shared compiler-side validator runs from
`load_graph()`, verifies every listed checksum, and requires the parameter,
input, and reference inventories to match the file-backed tensors in
`graph.json`. Graph-only fixtures remain supported. An older partial manifest
is verified as far as its inventory permits and produces an explicit warning.

## Binary File Format

All binary files are raw arrays with no header.

All values are stored in little-endian machine order where that matters. For
int8 files, this is just byte order.

| Tensor kind | Dtype | Layout | File contents |
| --- | --- | --- | --- |
| Activations | signed int8 | `CHW` | `C * H * W` bytes |
| Conv weights | signed int8 | `OIHW` | `Cout * Cin * Kh * Kw` bytes |
| Linear weights | signed int8 | `OI` | `Cout * Cin` bytes |
| Biases | signed int8 | `O` | `Cout` bytes |
| References | signed int8 | `CHW` or `C` | output tensor bytes |

Biases must be exported as signed int8. DeepTile will widen each bias lane to
int32 and apply the hardware bias alignment shift internally before writing the
bias buffer consumed by the kernel.

## Graph JSON Overview

The model handoff file must be named `graph.json` unless the DeepTile importer is
given another path explicitly.

Top-level structure:

```json
{
  "schema": "deeptile.graph.v1",
  "model": {},
  "target": {},
  "graph": {},
  "tensors": {},
  "nodes": []
}
```

Required top-level fields:

- `schema`: `"deeptile.graph.v1"`; the former `"tilecnn.graph.v1"` is also accepted.
- `model`: descriptive metadata.
- `target`: global assumptions for the exported graph.
- `graph`: names of graph inputs, outputs, and optional output references.
- `tensors`: dictionary of all graph, parameter, input, and reference tensors.
- `nodes`: topologically ordered operation list.

`graph.inputs` may contain more than one tensor. This is useful for exported
subgraphs that start at a residual block or a tail section and therefore need
multiple externally supplied activation tensors.

## Top-Level Metadata

Example:

```json
{
  "schema": "deeptile.graph.v1",
  "model": {
    "name": "resnet18",
    "producer": "fixed_point_quantizer",
    "producer_version": "0.1.0",
    "source_framework": "pytorch"
  },
  "target": {
    "bitwidth": 8,
    "signed": true,
    "activation_layout": "CHW",
    "conv_weight_layout": "OIHW",
    "linear_weight_layout": "OI",
    "bias_dtype": "int8"
  },
  "graph": {
    "inputs": ["input"],
    "outputs": ["logits"],
    "references": {
      "logits": "logits_ref"
    }
  },
  "tensors": {},
  "nodes": []
}
```

Target constraints for `deeptile.graph.v1`:

- `bitwidth` must be `8`.
- `signed` must be `true`.
- `activation_layout` must be `"CHW"`.
- `conv_weight_layout` must be `"OIHW"`.
- `linear_weight_layout` must be `"OI"`.
- `bias_dtype` must be `"int8"`.

The `target` object describes the canonical exported tensor format, not the
hardware execution format. It should not include DeepTile channel-packing fields
such as `IC_TILE_SIZE`, `OC_TILE_SIZE`, `channel_pack_in`, or
`channel_pack_out`. DeepTile obtains those values from its own build-time
configuration and uses them while packing tensors and generating schedules.

## Tensor Records

All tensors referenced by nodes must appear in the `tensors` dictionary.

Each int8 tensor has exactly one fixed-point fractional-bit value named `frac`.
The exporter should not separately write `frac_in` and `frac_out` fields on
nodes. Those names are DeepTile's per-launch view of the same tensor metadata:

```text
frac_in  = tensors[node.inputs.ifm].frac
frac_w   = tensors[node.inputs.weight].frac
frac_b   = tensors[node.inputs.bias].frac
frac_out = tensors[node.outputs.ofm].frac
```

For a tensor produced by one node and consumed by another, the producer's
`frac_out` and the consumer's `frac_in` are therefore the same tensor record:
`tensors[tensor_id].frac`.

This avoids duplicated quantization metadata and prevents inconsistencies where
one node claims a different output format than the next node's input format.

### Activation/Input Tensor

```json
"input": {
  "kind": "input",
  "dtype": "int8",
  "layout": "CHW",
  "shape": [3, 224, 224],
  "frac": 7,
  "file": "inputs/input_0.int8.bin"
}
```

Fields:

- `kind`: `"input"`, `"activation"`, `"param"`, or `"reference"`.
- `dtype`: `"int8"` for this schema version.
- `layout`: `"CHW"` for activations.
- `shape`: `[C, H, W]`.
- `frac`: fixed-point fractional bits for this tensor.
- `file`: relative path, required for file-backed inputs and references.

Intermediate activation tensors do not need a file unless they are exported as
debug references.

```json
"layer1.0.out": {
  "kind": "activation",
  "dtype": "int8",
  "layout": "CHW",
  "shape": [64, 56, 56],
  "frac": 5
}
```

### Convolution Weight Tensor

```json
"layer1.0.conv1_w": {
  "kind": "param",
  "dtype": "int8",
  "layout": "OIHW",
  "shape": [64, 64, 3, 3],
  "frac": 8,
  "file": "params/layer1.0.conv1.weight.int8.bin"
}
```

Shape is `[Cout, Cin, Kh, Kw]`.

### Linear Weight Tensor

```json
"fc_w": {
  "kind": "param",
  "dtype": "int8",
  "layout": "OI",
  "shape": [1000, 512],
  "frac": 8,
  "file": "params/fc.weight.int8.bin"
}
```

Shape is `[Cout, Cin]`. DeepTile lowers this to a `1x1` convolution with input
shape `[Cin, 1, 1]` and output shape `[Cout, 1, 1]`.

### Bias Tensor

```json
"layer1.0.conv1_b": {
  "kind": "param",
  "dtype": "int8",
  "layout": "O",
  "shape": [64],
  "frac": 6,
  "file": "params/layer1.0.conv1.bias.int8.bin"
}
```

Biases are exported as signed int8. DeepTile widens to int32 and computes the
hardware-aligned bias value using:

```text
shift_bias = frac_out - frac_b + 1
```

If `shift_bias >= 0`, DeepTile left-shifts the widened bias. If `shift_bias < 0`,
DeepTile right-shifts it.

### Reference Tensor

```json
"logits_ref": {
  "kind": "reference",
  "dtype": "int8",
  "layout": "CHW",
  "shape": [1000, 1, 1],
  "frac": 2,
  "file": "refs/logits.int8.bin"
}
```

References are optional but strongly recommended. At minimum, provide a final
graph output reference for bring-up. Intermediate references are useful for
debugging graph scheduler and residual ordering issues.

Final output references should be connected through `graph.references`, where
each key is a graph output tensor and each value is a tensor record with
`kind: "reference"`.

## Node Ordering

The `nodes` array must be in topological execution order.

Rules:

- A node may only consume tensors produced by earlier nodes, graph inputs, or
  parameter tensors.
- Residual add nodes must appear after both the main input and residual input
  tensors are available.
- Projection shortcut convolutions must appear before the conv node that uses
  their output as a residual input.
- The first DeepTile graph runtime will execute nodes sequentially in this order.

Parallel branch scheduling may be added later, but this schema does not require
the exporter to encode parallelism.

## Graph Nodes vs DeepTile Launches

The JSON file describes the model as a semantic graph. DeepTile then lowers that
semantic graph into one or more hardware launches.

Do not use the JSON `nodes` array to describe DeepTile's final fused launch plan.
For example, a ResNet stem should be exported as:

```text
conv2d -> maxpool2d
```

not as a single pre-fused `conv2d_maxpool2d` node.

DeepTile's graph compiler is responsible for recognizing legal patterns and
fusing them into a single `tile_conv` launch when the current hardware supports
that fusion.

This distinction gives us three benefits:

- the exported graph remains close to the original neural network semantics
- unfused or future fallback implementations remain possible
- validation can compare either semantic node outputs or fused launch outputs

In this document:

- **graph node** means an operation exported in `graph.json`
- **DeepTile launch** means one invocation of the `tile_conv` hardware kernel
- **fusion** means DeepTile combines multiple graph nodes into one launch

Residual add is the deliberate v1 exception. Instead of exporting a standalone
`add` graph node, the exporter should attach the residual input and residual
post-op flags to the terminal `conv2d` node of a residual block. This matches
the current DeepTile kernel contract, where residual loading and addition are
part of the convolution epilogue.

## Export Schema vs DeepTile Scheduling

The exporter should not attempt to describe DeepTile's internal tiling strategy.
In particular, `graph.json` should not contain:

- spatial tile sizes
- channel tile sizes
- descriptor counts
- weight-transfer epochs
- packed tensor strides
- hardware buffer-depth decisions

Those are generated by the DeepTile graph compiler after import.

For example, a `conv2d` node with a `7x7` kernel or stride `2` is valid if the
operation itself is supported. If a default DeepTile tile shape would exceed an
internal buffer, the DeepTile scheduler must choose a smaller legal tile and emit
more descriptors. That is not an exporter/schema error.

## Supported Ops in Schema v1

The version 1 graph importer supports:

- `conv2d`
- `maxpool2d`
- `gap2d`
- `linear`

DeepTile lowering rules:

| Graph pattern | DeepTile lowering |
| --- | --- |
| `conv2d` | one `tile_conv` launch |
| `conv2d` with ReLU | `tile_conv` with `RELU_ON` |
| `conv2d` with residual add | `tile_conv` with `ADD_ENABLE` |
| `conv2d` with residual add and ReLU | `ADD_ENABLE` plus `POST_ADD_RELU` |
| `conv2d` with ReLU6 | `tile_conv` with the ReLU6 clamp |
| dense `conv2d` followed by an eligible 3x3 or 5x5 depthwise `conv2d` | one `tile_conv` launch with the fused depthwise stage, when fusion is enabled |
| other depthwise `conv2d` | one `tile_conv` launch as a diagonal dense convolution |
| semantic `conv2d` followed by fusible `maxpool2d` | one `tile_conv` launch with `POOL_ENABLE` |
| semantic `conv2d` followed by fusible `gap2d` | one `tile_conv` launch with `GAP_ENABLE` |
| `linear` | one `tile_conv` launch as `1x1` convolution |

For v1, standalone non-fused maxpool and standalone non-fused GAP may be rejected
by the DeepTile importer unless a software or hardware fallback is implemented.

## Conv2D Node

Example:

```json
{
  "id": "layer1.0.conv1",
  "op": "conv2d",
  "inputs": {
    "ifm": "conv1_pool_out",
    "weight": "layer1.0.conv1_w",
    "bias": "layer1.0.conv1_b"
  },
  "outputs": {
    "ofm": "layer1.0.conv1_out"
  },
  "attrs": {
    "kernel": [3, 3],
    "stride": [1, 1],
    "padding": [1, 1, 1, 1],
    "dilation": [1, 1],
    "groups": 1
  },
  "post_ops": {
    "relu": true
  }
}
```

Quantization is primarily read from the input, weight, bias, and output tensor
records. DeepTile maps tensor-level `frac` values into its current per-layer
fields:

```text
frac_in  = tensors[inputs.ifm].frac
frac_w   = tensors[inputs.weight].frac
frac_b   = tensors[inputs.bias].frac
frac_out = tensors[outputs.ofm].frac
```

DeepTile derives:

```text
shift_out  = (frac_in + frac_w) - frac_out
shift_bias = frac_out - frac_b + 1
```

Current DeepTile constraints:

- `groups` must be `1` (dense convolution) or equal to both `Cin` and `Cout`
  (depthwise convolution, MobileNet — weight shape `[C, 1, Kh, Kw]`). Any other
  grouped convolution (`1 < groups < Cin`) is unsupported. Eligible
  pointwise-to-depthwise pairs use the fused streaming stage; standalone or
  ineligible depthwise layers use diagonal-dense lowering as the correctness
  fallback (see [Hardware architecture](../concepts/hardware-architecture.md#depthwise-separable-layers)).
- `dilation` must be `[1, 1]`.
- `kernel[0] == kernel[1]`.
- `stride[0] == stride[1]`.
- padding should be symmetric for current runtime compatibility.

For `padding`, use `[top, bottom, left, right]`. For symmetric padding, this is
normally `[p, p, p, p]`.

The exporter does not need to consider whether a layer fits a particular
DeepTile tile shape. DeepTile's scheduler is responsible for selecting spatial
tiles that fit the current kernel buffers.

## Post-Operations

A node's `post_ops` object lists the operations fused onto its output. Every key
must be one of the following; the importer rejects any other key, and any value
that is not a boolean or an integer.

| Key | Type | Meaning |
|---|---|---|
| `relu` | boolean | Clamp the output below at 0 |
| `relu6` | boolean | Clamp the output to `[0, min(127, round(6 * 2^frac_out))]` |
| `residual_add` | boolean | Add the `residual` input before any post-add activation |
| `residual_shift` | integer | Residual alignment shift; defaults to `frac_out - frac_residual` |
| `post_add_relu` | boolean | Clamp the sum below at 0 |
| `post_add_relu6` | boolean | Clamp the sum to the ReLU6 range above |

Both clamps apply after saturation to signed 8-bit. `relu` and `relu6` are
exclusive, as are `post_add_relu` and `post_add_relu6`. A node with a residual
add applies its activation to the sum, not to the convolution result.

## Residual Add

Residual addition should be represented as a fused post-op on the terminal
convolution of the block.

Example identity block terminal conv:

```json
{
  "id": "layer1.0.conv2_add",
  "op": "conv2d",
  "inputs": {
    "ifm": "layer1.0.conv1_out",
    "weight": "layer1.0.conv2_w",
    "bias": "layer1.0.conv2_b",
    "residual": "conv1_pool_out"
  },
  "outputs": {
    "ofm": "layer1.0.out"
  },
  "attrs": {
    "kernel": [3, 3],
    "stride": [1, 1],
    "padding": [1, 1, 1, 1],
    "dilation": [1, 1],
    "groups": 1
  },
  "post_ops": {
    "residual_add": true,
    "post_add_relu": true
  }
}
```

DeepTile derives residual alignment from tensor fractional bits:

```text
residual_shift = frac_out - frac_residual
```

where:

```text
frac_out      = tensors[outputs.ofm].frac
frac_residual = tensors[inputs.residual].frac
```

If the quantization tool uses a different residual alignment rule, it may
optionally provide an explicit override:

```json
"post_ops": {
  "residual_add": true,
  "residual_shift": 0,
  "post_add_relu": true
}
```

If present, DeepTile should verify that the override is consistent with the
tensor fractional bits or emit a warning.

Residual shape requirements:

- residual tensor shape must match the output tensor shape
- residual tensor dtype must be signed int8
- residual tensor layout must be `CHW`

## Projection Shortcuts

Projection shortcuts should be represented as normal `conv2d` nodes whose output
is later consumed as the residual input.

Example:

```json
{
  "id": "layer2.0.downsample.0",
  "op": "conv2d",
  "inputs": {
    "ifm": "layer1.out",
    "weight": "layer2.0.downsample.0_w",
    "bias": "layer2.0.downsample.0_b"
  },
  "outputs": {
    "ofm": "layer2.0.skip"
  },
  "attrs": {
    "kernel": [1, 1],
    "stride": [2, 2],
    "padding": [0, 0, 0, 0],
    "dilation": [1, 1],
    "groups": 1
  },
  "post_ops": {
    "relu": false
  }
}
```

Then the terminal block convolution consumes `"layer2.0.skip"` as its residual.

## MaxPool2D Node

Example:

```json
{
  "id": "maxpool",
  "op": "maxpool2d",
  "inputs": {
    "ifm": "conv1_out"
  },
  "outputs": {
    "ofm": "conv1_pool_out"
  },
  "attrs": {
    "kernel": [3, 3],
    "stride": [2, 2],
    "padding": [1, 1, 1, 1]
  }
}
```

Current DeepTile maxpool support is specialized for:

- kernel `[3, 3]`
- stride `[2, 2]`
- padding `[1, 1, 1, 1]`

Preferred lowering:

```text
conv2d -> maxpool2d
```

The exporter should write this as two semantic graph nodes. DeepTile may lower
those two nodes into one hardware launch if:

- the maxpool input is the conv output
- no other node consumes the intermediate conv output
- quantization format is unchanged or representable by `post_pool_shift`

If a maxpool cannot be fused, the v1 DeepTile runtime may reject the graph.

## GAP2D Node

Example:

```json
{
  "id": "avgpool",
  "op": "gap2d",
  "inputs": {
    "ifm": "layer4.out"
  },
  "outputs": {
    "ofm": "gap_out"
  }
}
```

Current DeepTile GAP support is intended to be fused with the preceding
convolution when the full convolution output fits in one spatial tile.

The exporter should write GAP as a separate semantic `gap2d` node. DeepTile may
fuse it with the immediately preceding `conv2d` during lowering.

Whether the fused GAP pattern is legal for the current hardware is a DeepTile
compiler decision. The exporter only needs to provide the semantic `gap2d`
node, input/output tensor shapes, and tensor fractional bits.

DeepTile derives:

```text
gap_mul = round(2^(GAP_SCALE_FRAC_BITS + frac_out_gap - frac_out_conv) / (H * W))
```

where:

```text
frac_out_conv = tensors[inputs.ifm].frac
frac_out_gap  = tensors[outputs.ofm].frac
H * W         = spatial size of the GAP input
```

For schema v1, a non-fused `gap2d` may be rejected unless a DeepTile fallback is
implemented.

## Linear Node

Example:

```json
{
  "id": "fc",
  "op": "linear",
  "inputs": {
    "ifm": "gap_out",
    "weight": "fc_w",
    "bias": "fc_b"
  },
  "outputs": {
    "ofm": "logits"
  }
}
```

Input tensor shape should be `[Cin, 1, 1]`.

Weight tensor shape should be `[Cout, Cin]` with layout `OI`.

Output tensor shape should be `[Cout, 1, 1]`.

DeepTile lowers `linear` into a `1x1` convolution. This path is a compatibility
path for end-to-end ResNet execution, not an optimized classifier backend.

## Full Minimal Example

```json
{
  "schema": "deeptile.graph.v1",
  "model": {
    "name": "tiny_residual_tail",
    "producer": "fixed_point_quantizer",
    "producer_version": "0.1.0",
    "source_framework": "pytorch"
  },
  "target": {
    "bitwidth": 8,
    "signed": true,
    "activation_layout": "CHW",
    "conv_weight_layout": "OIHW",
    "linear_weight_layout": "OI",
    "bias_dtype": "int8"
  },
  "graph": {
    "inputs": ["input"],
    "outputs": ["block_out"]
  },
  "tensors": {
    "input": {
      "kind": "input",
      "dtype": "int8",
      "layout": "CHW",
      "shape": [64, 56, 56],
      "frac": 5,
      "file": "inputs/input_0.int8.bin"
    },
    "conv1_w": {
      "kind": "param",
      "dtype": "int8",
      "layout": "OIHW",
      "shape": [64, 64, 3, 3],
      "frac": 8,
      "file": "params/conv1.weight.int8.bin"
    },
    "conv1_b": {
      "kind": "param",
      "dtype": "int8",
      "layout": "O",
      "shape": [64],
      "frac": 3,
      "file": "params/conv1.bias.int8.bin"
    },
    "conv1_out": {
      "kind": "activation",
      "dtype": "int8",
      "layout": "CHW",
      "shape": [64, 56, 56],
      "frac": 5
    },
    "conv2_w": {
      "kind": "param",
      "dtype": "int8",
      "layout": "OIHW",
      "shape": [64, 64, 3, 3],
      "frac": 8,
      "file": "params/conv2.weight.int8.bin"
    },
    "conv2_b": {
      "kind": "param",
      "dtype": "int8",
      "layout": "O",
      "shape": [64],
      "frac": 3,
      "file": "params/conv2.bias.int8.bin"
    },
    "block_out": {
      "kind": "activation",
      "dtype": "int8",
      "layout": "CHW",
      "shape": [64, 56, 56],
      "frac": 5
    }
  },
  "nodes": [
    {
      "id": "conv1",
      "op": "conv2d",
      "inputs": {
        "ifm": "input",
        "weight": "conv1_w",
        "bias": "conv1_b"
      },
      "outputs": {
        "ofm": "conv1_out"
      },
      "attrs": {
        "kernel": [3, 3],
        "stride": [1, 1],
        "padding": [1, 1, 1, 1],
        "dilation": [1, 1],
        "groups": 1
      },
      "post_ops": {
        "relu": true
      }
    },
    {
      "id": "conv2_add",
      "op": "conv2d",
      "inputs": {
        "ifm": "conv1_out",
        "weight": "conv2_w",
        "bias": "conv2_b",
        "residual": "input"
      },
      "outputs": {
        "ofm": "block_out"
      },
      "attrs": {
        "kernel": [3, 3],
        "stride": [1, 1],
        "padding": [1, 1, 1, 1],
        "dilation": [1, 1],
        "groups": 1
      },
      "post_ops": {
        "residual_add": true,
        "post_add_relu": true
      }
    }
  ]
}
```

## Validation Requirements

The exporter should validate before writing files:

- all node inputs and outputs reference known tensors
- `nodes` are topologically ordered
- every parameter tensor has a file
- every file has exactly the expected byte count
- conv weight shape matches conv attrs and input/output channels
- linear weight shape matches input/output channels
- bias shape matches output channels
- residual input shape matches node output shape
- all `frac` values are present
- all dtypes are supported by this schema
- all operation-level DeepTile constraints are respected, such as supported ops,
  `groups=1`, `dilation=[1,1]`, square kernels, equal strides, and currently
  supported pooling attributes

Expected byte counts:

```text
activation CHW int8: C * H * W
conv weight OIHW int8: Cout * Cin * Kh * Kw
linear weight OI int8: Cout * Cin
bias O int8: Cout
reference CHW int8: C * H * W
```

## DeepTile importer behavior

The DeepTile graph importer:

1. Parse `graph.json`.
2. Validate schema, tensor records, node order, and file sizes.
3. Load file-backed graph inputs, params, and references.
4. Lower fusible graph patterns into per-layer `tile_conv` launch plans.
5. Choose legal tile shapes for each lowered launch.
6. Pack IFM tensors as needed.
7. Pack weights into the accelerator's weight-bank layout.
8. Widen and shift int8 biases into the existing `biases_vector_dt` format.
9. Generate `TileSchedule` for each launch.
10. Allocate activation, residual, descriptor, weight, and bias buffers.
11. Run the graph sequentially.
12. Compare final output and optional intermediate outputs against references.

## Versioning notes

This is schema version `deeptile.graph.v1`.

The following capabilities are outside schema version 1:

- packed tensor export
- single archive files with byte offsets
- grouped convolution other than depthwise (`1 < groups < C`)
- non-symmetric padding
- standalone pooling
- standalone GAP
- dedicated classifier/GEMV backend
- per-tensor scale/zero-point quantization metadata
- dynamic runtime inputs without file-backed activation data

Version 1 keeps the contract small, explicit, and close to the accelerator's
execution model.
