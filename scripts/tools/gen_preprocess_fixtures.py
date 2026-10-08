#!/usr/bin/env python3
"""Generate host preprocessing test fixtures from Pillow.

Pillow is the ground truth for the exporter's resize, so expected outputs are
produced here and checked in. The C++ implementation in src/host/dataset/ must
reproduce them bit-exactly. Regenerate only when a case is added; changing an
existing expectation means the contract moved and needs justifying.

Test inputs are not checked in. They are synthesized by the LCG below, which
tests/host/preprocess_tests.cpp mirrors exactly, so only expected outputs
occupy the repository.

Usage: python3 scripts/tools/gen_preprocess_fixtures.py
Requires: Pillow, numpy.
"""

import pathlib
import numpy as np
from PIL import Image

OUT = pathlib.Path(__file__).resolve().parents[2] / "tests/host/data/preprocess"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def synth(width, height, seed):
    """Gradient plus LCG noise. Must match synth_image() in the C++ test."""
    out = np.empty((height, width, 3), np.uint8)
    state = seed & 0xFFFFFFFF
    for y in range(height):
        for x in range(width):
            for c in range(3):
                state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
                noise = ((state >> 16) & 0x3F) - 32
                if c == 0:
                    base = 255 * x // max(width - 1, 1)
                elif c == 1:
                    base = 255 * y // max(height - 1, 1)
                else:
                    base = 255 * ((x // 3 + y // 5) % 2)
                out[y, x, c] = min(255, max(0, base + noise))
    return out


def eval_size(w, h, short):
    """torchvision Resize(int): truncating, not rounding."""
    if w < h:
        return short, int(short * h / w)
    return int(short * w / h), short


def write_ppm(path, arr):
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (arr.shape[1], arr.shape[0]))
        f.write(arr.tobytes())


def quantize(rgb):
    x = (np.asarray(rgb, np.float32) / 255.0 - MEAN) / STD
    x = np.transpose(x, (2, 0, 1))
    return np.clip(np.floor(x * 32 + 0.5), -128, 127).astype(np.int8)


# name, width, height, seed, out_width, out_height
RESIZE_CASES = [
    ("downscale", 97, 61, 11, 40, 24),
    ("upscale", 17, 23, 22, 40, 40),
    ("width_only", 64, 48, 33, 32, 48),
    ("passthrough", 33, 21, 44, 33, 21),
    ("odd_ratio", 123, 45, 55, 37, 29),
]

CONVENTION = ("convention", 400, 300, 99)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, w, h, seed, ow, oh in RESIZE_CASES:
        src = synth(w, h, seed)
        write_ppm(OUT / f"{name}_out.ppm",
                  np.asarray(Image.fromarray(src).resize((ow, oh), Image.BILINEAR)))

    name, w, h, seed = CONVENTION
    img = Image.fromarray(synth(w, h, seed))
    quantize(img.resize((224, 224), Image.BILINEAR)).tofile(
        OUT / f"{name}_export.int8.bin")

    ow, oh = eval_size(w, h, 256)
    resized = img.resize((ow, oh), Image.BILINEAR)
    left, top = (ow - 224) // 2, (oh - 224) // 2
    quantize(resized.crop((left, top, left + 224, top + 224))).tofile(
        OUT / f"{name}_eval.int8.bin")

    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
