#!/usr/bin/env python3
"""Derive a DeepTile evaluation root from a PyTorch ImageFolder validation split.

The source tree (for example ``imagenet-mini/val``) is shared with other
consumers and is never modified: this script only reads it. Images are linked
or copied into the flat layout ``load_imagenet_index`` expects, and the emitted
``val.txt`` lists exactly the images that were placed, so a short or damaged
evaluation set cannot be mistaken for a complete one.

Every filename is checked against the official ILSVRC2012 validation list and
every source synset directory is checked against the official label. A single
disagreement aborts the run: a subset whose labels are not the official ones is
not a subset of ImageNet, and silently evaluating against it would produce an
accuracy number with no meaning.

Typical use::

    scripts/data/prepare_imagenet_subset.py \\
        --source ~/Documents/datasets/imagenet-mini/val \\
        --out data/imagenet

Add ``--copy`` when staging for a board that cannot follow the symlinks. The
same effect is available from the default symlink farm with ``tar -h``, which
avoids a second copy on the host.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

IMAGE_SUBDIR = Path("ilsvrc2012") / "ILSVRC2012_img_val"
OFFICIAL_LIST = "ilsvrc2012_val_full.txt"
SYNSET_WORDS = "synset_words.txt"


def die(message: str) -> "NoReturn":  # type: ignore[valid-type]
    sys.stderr.write(f"error: {message}\n")
    raise SystemExit(1)


def load_official_labels(path: Path) -> dict[str, int]:
    """Read "<name> <class-index>" lines into a lookup."""
    labels: dict[str, int] = {}
    with path.open() as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 2:
                die(f"{path}:{lineno}: expected '<name> <label>', got {line!r}")
            labels[fields[0]] = int(fields[1])
    if not labels:
        die(f"{path} contains no entries")
    return labels


def load_synset_order(path: Path) -> dict[str, int]:
    """Map each synset id to its class index by position in synset_words.txt."""
    order: dict[str, int] = {}
    with path.open() as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if line:
                order[line.split()[0]] = index
    if not order:
        die(f"{path} contains no entries")
    return order


def collect_source_images(source: Path, synset_order: dict[str, int]) -> list[tuple[Path, int]]:
    """List (path, expected-label) for every image in an ImageFolder tree."""
    found: list[tuple[Path, int]] = []
    for synset_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        expected = synset_order.get(synset_dir.name)
        if expected is None:
            die(f"{synset_dir.name} is not a known synset; is {source} an "
                "ImageNet validation split?")
        for image in synset_dir.iterdir():
            if image.is_file():
                found.append((image, expected))
    if not found:
        die(f"no images found under {source}")
    return found


def verify(found: list[tuple[Path, int]], official: dict[str, int]) -> list[str]:
    """Check every image against the official list; return unknown filenames."""
    unknown: list[str] = []
    mismatched: list[str] = []
    for image, expected in found:
        actual = official.get(image.name)
        if actual is None:
            unknown.append(image.name)
        elif actual != expected:
            mismatched.append(
                f"{image.name}: source says {expected}, official list says {actual}")
    if mismatched:
        preview = "\n  ".join(mismatched[:10])
        more = "" if len(mismatched) <= 10 else f"\n  ... and {len(mismatched) - 10} more"
        die(f"{len(mismatched)} label disagreements with the official list:\n"
            f"  {preview}{more}")
    return unknown


def place(image: Path, destination: Path, copy: bool) -> None:
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    if copy:
        shutil.copy2(image, destination)
    else:
        destination.symlink_to(image.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive a DeepTile evaluation root from an ImageFolder split.")
    parser.add_argument("--source", required=True, type=Path,
                        help="ImageFolder validation split, e.g. imagenet-mini/val")
    parser.add_argument("--out", required=True, type=Path,
                        help="DeepTile dataset root to populate (e.g. data/imagenet)")
    parser.add_argument("--copy", action="store_true",
                        help="copy images instead of symlinking (for board staging)")
    parser.add_argument("--allow-unknown", action="store_true",
                        help="skip images absent from the official list instead of failing")
    args = parser.parse_args()

    source: Path = args.source
    out: Path = args.out
    if not source.is_dir():
        die(f"source is not a directory: {source}")
    if not out.is_dir():
        die(f"dataset root does not exist: {out}")

    official_path = out / OFFICIAL_LIST
    synset_path = out / SYNSET_WORDS
    for required in (official_path, synset_path):
        if not required.is_file():
            die(f"missing {required}; it is tracked in the repository")

    official = load_official_labels(official_path)
    synset_order = load_synset_order(synset_path)

    found = collect_source_images(source, synset_order)
    unknown = verify(found, official)
    if unknown and not args.allow_unknown:
        preview = ", ".join(unknown[:5])
        die(f"{len(unknown)} images are not in the official validation list "
            f"({preview}...); pass --allow-unknown to skip them")
    if unknown:
        skip = set(unknown)
        found = [(p, label) for p, label in found if p.name not in skip]
        print(f"skipped {len(unknown)} images absent from the official list")

    # Official order keeps the emitted listing stable and independent of both
    # the source directory layout and the filesystem's iteration order.
    official_position = {name: i for i, name in enumerate(official)}
    found.sort(key=lambda item: official_position[item[0].name])

    image_dir = out / IMAGE_SUBDIR
    image_dir.mkdir(parents=True, exist_ok=True)
    stale = {p.name for p in image_dir.iterdir()} - {p.name for p, _ in found}
    for name in stale:
        (image_dir / name).unlink()
    if stale:
        print(f"removed {len(stale)} images no longer in the subset")

    for image, _ in found:
        place(image, image_dir / image.name, args.copy)

    listing = "".join(f"{image.name} {label}\n" for image, label in found)
    (out / "val.txt").write_text(listing)

    digest = hashlib.sha256(listing.encode()).hexdigest()
    classes = len({label for _, label in found})
    print(f"prepared {len(found)} images from {classes} classes in {image_dir}")
    print(f"mode: {'copied' if args.copy else 'symlinked'}")
    print(f"val.txt sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
