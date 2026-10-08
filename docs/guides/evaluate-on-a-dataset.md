# Evaluate on a dataset

Dataset mode runs a whole labeled image set through the accelerator and reports
top-1 and top-k accuracy alongside throughput, from a single run. This guide
prepares an ImageNet evaluation set and runs it on the board.

## 1. Prepare the images

DeepTile reads an ImageNet-style directory with a label listing:

```text
data/imagenet/
|-- val.txt                             # the evaluation set: "<file> <class-index>"
|-- synset_words.txt                    # 1,000 class names, sorted-synset order
|-- ilsvrc2012_val_full.txt             # official 50,000-image validation listing
`-- ilsvrc2012/ILSVRC2012_img_val/      # the images (not tracked by Git)
```

The listings are tracked in the repository; the images are not
redistributed by this repository. The paper uses the validation split of
[ImageNet 1000 (mini)](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000)
by Ilya Figotin. Point the preparation script at a PyTorch `ImageFolder`
validation directory (`val/<synset>/<file>`):

```bash
scripts/data/prepare_imagenet_subset.py \
    --source /path/to/imagenet/val \
    --out data/imagenet
```

The script symlinks the images rather than copying them, checks every file name
and label against the official listing, and prints the SHA-256 of the `val.txt`
it writes. That digest identifies exactly which images a run evaluated. Add
`--copy` to write real files instead of symlinks.

!!! note "Every listed image must exist"
    `val.txt` defines the evaluation set, so a missing image is an error:
    accuracy over an unidentified subset would be meaningless. Use
    `--allow-missing` only for smoke tests.

## 2. Stage the images on the board

```bash
tar czhf - -C data/imagenet ilsvrc2012 | ssh zcu104 'tar xzf - -C ~/boot/data/imagenet'
```

`h` makes `tar` follow the symlinks. Copy the listings the same way if the
board does not have them. Target deployments never overwrite the board's
`data/` directory, so you only stage a dataset once.

## 3. Run the evaluation

On the board, from the boot partition (see [Run a model](run-a-model.md#4-run-it)):

```bash
./deeptile target.json models/resnet50/imagenet1k/int8-tqt/v1.0.0/graph.json \
    --dataset data/imagenet --topk 5 --output results/resnet50-imagenet
```

For MobileNetV2 and EfficientNet-Lite0, prefix the command with
`DEEPTILE_FUSE_DW=1`.

The host prints top-1 and top-5 accuracy, accelerator time, preparation cost,
and the fraction of wall time the accelerator was busy. With `--output` it
writes `run.json` and `predictions.csv`, the top-k predictions for every image
in listing order.

| Option | Effect |
|---|---|
| `--dataset ROOT` | Dataset root; selects dataset mode |
| `--topk N` | Report top-k alongside top-1 (default 5) |
| `--limit N` | Evaluate at most N images |
| `--workers N` | Preprocessing threads (default 2) |
| `--queue N` | Prepared images buffered (default twice the workers) |
| `--convention NAME` | `eval` (default) or `export` |
| `--short-side N` | Resize target for `eval` (default 256) |
| `--allow-missing` | Continue when `val.txt` names absent images |

## Preprocessing

The host reproduces the standard ImageNet evaluation transform in integer
arithmetic: decode as RGB, resize the shorter side to 256 pixels with a
Pillow-compatible bilinear filter, centre-crop to 224 × 224, normalize with the
ImageNet mean and standard deviation, and quantize to the model's signed 8-bit
input format. For a network with a folded stem (see
[Execution and timing](../concepts/execution-and-timing.md#stem-folding)), the
prepared tensor is written directly in the folded layout.

| Convention | Transform | Use |
|---|---|---|
| `eval` (default) | Shorter side to 256, centre-crop 224 | Accuracy measurement |
| `export` | Direct resize to 224 × 224 | Reproducing FixQuant's packaged validation input |

The resize is DeepTile's own implementation, bit-identical to Pillow and
vectorized with NEON on the board, because no OpenCV resize mode reproduces
Pillow closely enough to attribute a prediction difference to the hardware
rather than to preprocessing. When the host is built with libjpeg, as the board
host is, JPEG files are decoded only over the region the crop needs; otherwise
OpenCV decodes the whole image. Both paths produce identical tensors.

## Choosing workers

Preparing an image on the ZCU104's CPU costs about as much as running it, so
preprocessing runs on worker threads in parallel with inference. Predictions are
identical at any worker count; only throughput changes.

The reported evaluation uses three workers and a queue depth of six.
Networks with short inference, such as MobileNetV2 and EfficientNet-Lite0, are
limited by preparation and benefit from more workers — but more workers also
compete with the thread that drives the accelerator, which inflates the
kernel-time figure. See
[Execution and timing](../concepts/execution-and-timing.md#cpu-contention)
before quoting accelerator throughput from a dataset run.

To measure preprocessing cost on its own, without the accelerator:

```bash
./build/tools/dataset_preprocess data/imagenet --limit 100 --convention eval
```

## Interpreting the accuracy

- **Top-k ties use a fixed rule: the lowest class index wins.** Integer logits
  tie often, so without a rule top-k accuracy would not be reproducible.
- **Compare against a baseline on the same image list.** The repository's
  default set is 3,923 images from the ImageNet-mini validation split, covering
  all 1,000 classes. It is not a uniform sample of the full validation set, so
  its accuracy is not an unbiased estimate of full-validation accuracy. At this
  size the 95% interval on a top-1 near 76% is about ±1.3 percentage points.
- **Dataset accuracy measures the deployed model, not the hardware alone.**
  Hardware correctness is established separately, by exact comparison against
  the packaged integer references.
