#!/usr/bin/env python3
"""Export and validate a FixQuant model-zoo release for DeepTile."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


_RELEASE_ID = re.compile(
    r"(?P<model>[A-Za-z0-9._-]+)/"
    r"(?P<dataset>[A-Za-z0-9._-]+)/"
    r"(?P<profile>[A-Za-z0-9._-]+)@"
    r"(?P<version>[A-Za-z0-9._-]+)"
)


def release_path(release_id: str) -> Path:
    match = _RELEASE_ID.fullmatch(release_id)
    if match is None:
        raise ValueError(
            "release ID must be model/dataset/profile@version"
        )
    parts = tuple(match.group(name) for name in (
        "model", "dataset", "profile", "version"
    ))
    if any(part in {".", ".."} for part in parts):
        raise ValueError("release ID components cannot be '.' or '..'")
    return Path(*parts)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_package_artifact(package_dir: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(
            f"model-package artifact path is unsafe: {relative_path}"
        )
    resolved = (package_dir / path).resolve()
    try:
        resolved.relative_to(package_dir.resolve())
    except ValueError as exc:
        raise ValueError(
            f"model-package artifact escapes package: {relative_path}"
        ) from exc
    if not resolved.is_file():
        raise ValueError(f"model-package artifact is missing: {relative_path}")
    return resolved


def validate_artifact_record(package_dir: Path, record: object) -> str:
    if not isinstance(record, dict):
        raise ValueError("model-package artifact record must be an object")
    relative_path = record.get("path")
    expected_hash = record.get("sha256")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("model-package artifact path is missing")
    if not isinstance(expected_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_hash):
        raise ValueError(
            f"model-package artifact checksum is invalid: {relative_path}"
        )
    artifact = resolve_package_artifact(package_dir, relative_path)
    actual_hash = sha256_file(artifact)
    if actual_hash != expected_hash:
        raise ValueError(
            f"model-package artifact checksum mismatch: {relative_path}"
        )
    return relative_path


def validate_package_manifest(package_dir: Path, release_id: str) -> None:
    manifest_path = package_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    # tilecnn.* is the former namespace of the same format.
    if manifest.get("schema") not in ("deeptile.model-package.v1",
                                      "tilecnn.model-package.v1"):
        raise ValueError("unsupported or missing model-package schema")
    model = manifest.get("model")
    if not isinstance(model, dict) or model.get("release_id") != release_id:
        raise ValueError("model-package release ID does not match the request")

    integrity = manifest.get("integrity", {})
    if integrity != {
            "algorithm": "sha256",
            "coverage": "all-graph-artifacts"}:
        raise ValueError("model-package integrity coverage is incomplete")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("model-package artifacts object is missing")
    graph_record = artifacts.get("graph", {})
    if not isinstance(graph_record, dict) or graph_record.get("path") != "graph.json":
        raise ValueError("model-package graph path must be graph.json")
    validate_artifact_record(package_dir, graph_record)

    with (package_dir / "graph.json").open("r", encoding="utf-8") as handle:
        graph = json.load(handle)
    tensors = graph.get("tensors")
    if not isinstance(tensors, dict):
        raise ValueError("model-package graph tensors object is missing")

    category_for_kind = {
        "param": "parameters",
        "input": "inputs",
        "reference": "references",
    }
    expected = {category: set() for category in category_for_kind.values()}
    for tensor_id, tensor in tensors.items():
        if not isinstance(tensor, dict) or not tensor.get("file"):
            continue
        if not isinstance(tensor["file"], str):
            raise ValueError(
                f"file-backed tensor path must be a string: {tensor_id}"
            )
        kind = tensor.get("kind")
        if kind not in category_for_kind:
            raise ValueError(
                f"file-backed tensor has unsupported kind: {tensor_id}"
            )
        expected[category_for_kind[kind]].add(tensor["file"])

    seen = set()
    for category, expected_paths in expected.items():
        records = artifacts.get(category)
        if not isinstance(records, list):
            raise ValueError(
                f"model-package artifact list is missing: {category}"
            )
        actual_paths = set()
        for record in records:
            relative_path = validate_artifact_record(package_dir, record)
            if relative_path in seen:
                raise ValueError(
                    f"model-package artifact is listed more than once: "
                    f"{relative_path}"
                )
            seen.add(relative_path)
            actual_paths.add(relative_path)
        if actual_paths != expected_paths:
            missing = sorted(expected_paths - actual_paths)
            unexpected = sorted(actual_paths - expected_paths)
            raise ValueError(
                f"model-package {category} inventory mismatch; "
                f"missing={missing}, unexpected={unexpected}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zoo-model",
        required=True,
        help="Released model ID: model/dataset/profile@version",
    )
    parser.add_argument(
        "--zoo-root",
        help="Override the FixQuant model-zoo root",
    )
    parser.add_argument(
        "--image",
        help="Override the image used to generate the exported reference",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Model-package root; defaults to build/models",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = (args.output_root or repo_root / "build" / "models").resolve()

    try:
        relative_path = release_path(args.zoo_model)
    except ValueError as exc:
        print(f"model export: {exc}", file=sys.stderr)
        return 2

    exporter = repo_root / "FixQuant" / "tools" / "export_deeptile_graph.py"
    graph_plan = repo_root / "build" / "tools" / "graph_plan"
    if not exporter.is_file():
        print("model export: FixQuant submodule is not initialized", file=sys.stderr)
        return 2
    if not graph_plan.is_file():
        print(
            "model export: build graph_plan first with "
            "'make -C scripts graph_plan'",
            file=sys.stderr,
        )
        return 2

    final_dir = output_root / relative_path
    if final_dir.exists():
        try:
            validate_package_manifest(final_dir, args.zoo_model)
            validation = subprocess.run(
                [str(graph_plan), str(final_dir / "graph.json")],
                cwd=repo_root,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as exc:
            if exc.stdout:
                print(exc.stdout, file=sys.stderr)
            print(
                f"existing model-package validation failed with status "
                f"{exc.returncode}",
                file=sys.stderr,
            )
            return exc.returncode or 1
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(
                f"existing model-package validation failed: {exc}",
                file=sys.stderr,
            )
            return 2

        print(f"ModelPackage already exists and is valid: {final_dir}")
        print("Validated DeepTile graph plan:")
        for line in validation.stdout.splitlines()[:7]:
            print(f"  {line}")
        return 0

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{final_dir.name}.", dir=final_dir.parent)
    )

    export_command = [
        sys.executable,
        str(exporter),
        "--zoo-model",
        args.zoo_model,
        "--out_dir",
        str(staging_dir),
    ]
    if args.zoo_root:
        export_command.extend(["--zoo-root", args.zoo_root])
    if args.image:
        export_command.extend(["--image", args.image])

    try:
        print(f"Exporting {args.zoo_model}")
        subprocess.run(export_command, cwd=repo_root, check=True)
        validate_package_manifest(staging_dir, args.zoo_model)

        validation = subprocess.run(
            [str(graph_plan), str(staging_dir / "graph.json")],
            cwd=repo_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print("Validated DeepTile graph plan:")
        for line in validation.stdout.splitlines()[:7]:
            print(f"  {line}")

        staging_dir.rename(final_dir)
        print(f"ModelPackage ready: {final_dir}")
        return 0
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, file=sys.stderr)
        print(f"model export failed with status {exc.returncode}", file=sys.stderr)
        return exc.returncode or 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"model export failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


if __name__ == "__main__":
    raise SystemExit(main())
