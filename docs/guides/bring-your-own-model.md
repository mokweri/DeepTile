# Bring your own model

DeepTile runs any network that is exported as a ModelPackage and uses only the
operators the accelerator supports. There are three ways to get one.

| Route | When to use it |
|---|---|
| [Use a released FixQuant model](#use-a-released-model) | You want a validated network without training |
| [Train and export with FixQuant](#train-and-export-with-fixquant) | You have your own network or dataset |
| [Export from another quantizer](#export-from-another-quantizer) | You already quantize with a different tool |

In every case, first check that the network fits the
[supported operators](../reference/operators.md).

## Use a released model

FixQuant publishes versioned, checksummed models. Five are validated end to
end on DeepTile:

| Network | Release ID |
|---|---|
| ResNet-50 | `resnet50/imagenet1k/int8-tqt@v1.0.0` |
| ResNet-18 | `resnet18/imagenet1k/int8-tqt@v1.0.0` |
| MobileNetV2 | `mobilenet_v2/imagenet1k/int8-tqt-cle@v1.0.0` |
| EfficientNet-Lite0 | `efficientnet_lite0/imagenet1k/int8-tqt-cle-relu6@v1.0.0` |
| `vgg16_tilecnn` (VGG-class, not stock VGG-16) | `vgg16_tilecnn/imagenet1k/int8-tqt@v1.0.0` |

Fetch and export any of them by its release ID:

```bash
make -C scripts model-fetch  ZOO_MODEL=resnet50/imagenet1k/int8-tqt@v1.0.0
make -C scripts model-export ZOO_MODEL=resnet50/imagenet1k/int8-tqt@v1.0.0
```

`model-fetch` downloads the checkpoint once and verifies its size and SHA-256.
`model-export` exports into a temporary directory, verifies every file against
the package's checksum inventory, validates the graph with the DeepTile
compiler, and only then publishes the package under
`build/models/<model>/<dataset>/<profile>/<version>/`. Export never downloads
anything, so it cannot make an unexpected network request.

## Train and export with FixQuant

FixQuant trains quantization-aware models with 8-bit power-of-two scales and
exports the graph, tensor metadata, parameters, and optional integer reference
needed by DeepTile. Follow the
[FixQuant documentation](https://github.com/mokweri/FixQuantTool) to train or
prepare your network, then export it:

```bash
python FixQuant/tools/export_deeptile_graph.py \
    --zoo-model <model/dataset/profile@version> \
    --out_dir outputs/<model>_int8
```

Registering your model as a FixQuant release lets you use the `model-fetch` and
`model-export` targets above, with the same verification.

## Export from another quantizer

Any tool can produce a ModelPackage by following the
[graph handoff specification](../reference/graph-handoff-spec.md). A package is
a directory:

```text
<model>_int8/
|-- manifest.json    # identity, provenance, and checksums (recommended)
|-- graph.json       # graph topology, tensor records, and fixed-point metadata
|-- params/          # signed 8-bit weights and biases
|-- inputs/          # optional validation input
`-- refs/            # optional integer reference outputs for that input
```

Your exporter provides the graph, tensors in canonical layouts (`CHW`
activations, `OIHW` convolution weights, `OI` linear weights), and the
fractional bit count of every tensor. DeepTile does the rest: it packs tensors
for the accelerator, chooses tiles, derives the rescaling shifts, and generates
descriptors. Do not pre-pack tensors for DeepTile.

!!! tip "Always export a reference output"
    A package without reference outputs is a valid compiler input, but nothing
    can then confirm that the accelerator reproduces your quantized model. Export
    the integer output your quantizer computes for the packaged input, and DeepTile
    will check the hardware against it on every run.

## Check that the model fits

Before building anything, confirm that DeepTile accepts the graph, using the
native tools:

```bash
make -C scripts tools

# Prepare the full runtime plan, without a device
./build/tools/runtime_dry_run path/to/graph.json

# Show how each layer is lowered and scheduled
./build/tools/graph_schedule path/to/graph.json
```

To simulate the whole network on the kernel before running it on hardware:

```bash
make -C scripts hls-csim DEEPTILE_GRAPH=path/to/graph.json
```

C simulation compares the kernel's output against your exported reference, so
it is an exact correctness check. Whole-network simulation is slow; a
representative subset of layers is often enough while you iterate.

Once the graph passes, copy the package to the board and run it as in
[Run a model](run-a-model.md). An existing target runs it without a new FPGA
build.
