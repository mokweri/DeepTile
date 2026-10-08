# Naming

DeepTile was previously named TileCNN. Everything DeepTile and FixQuant build,
write and read now uses the new name. The former names are still accepted
wherever an existing script, record or package might use them, so nothing made
before the rename stops working.

## What changed

| Former | Current | Where |
|---|---|---|
| `tileCNN` | `deeptile` | Host executable inside every target package |
| `tileCNN.xclbin` | `deeptile.xclbin` | FPGA binary |
| `tilecnn.kernel` | `deeptile.kernel`, version 2 | Kernel ABI name in every target manifest |
| `TILECNN_*` | `DEEPTILE_*` | Environment, Make and HLS variables |
| `.tilecnn-deploy-boot-id` | `.deeptile-deploy-boot-id` | Reboot marker on the board's boot partition |
| `tilecnn.*.v1` | `deeptile.*.v1` | Record-format identifiers |
| `tilecnn_revision` | `deeptile_revision` | Source-revision field in target manifests, build records and RunRecords |

Kernel ABI version 2 also carries the ReLU6 clamp and the fused depthwise
kernel size (`dw_config` bit 4). A host accepts only targets
with its own ABI name and version, so a host from before the rename cannot drive
a current target, nor the reverse.

## What still accepts the former names

- **Environment variables.** When a `DEEPTILE_*` variable is unset, the host,
  compiler, testbench, Make and HLS flows read its `TILECNN_*` counterpart.
  `DEEPTILE_FUSE_DW=1` and `TILECNN_FUSE_DW=1` therefore behave the same.
- **Record formats.** Both identifiers name the same formats, and readers accept
  either:

  | Former identifier | Current identifier | Record |
  |---|---|---|
  | `tilecnn.graph.v1` | `deeptile.graph.v1` | Model graph |
  | `tilecnn.model-package.v1` | `deeptile.model-package.v1` | ModelPackage manifest |
  | `tilecnn.target.v1` | `deeptile.target.v1` | TargetPackage manifest |
  | `tilecnn.target-build.v1` | `deeptile.target-build.v1` | Target build record |
  | `tilecnn.run.v1` | `deeptile.run.v1` | RunRecord |

- **FixQuant.** `export_tilecnn_graph.py`, `check_tilecnn_legality.py`,
  `fixquant.export.tilecnn_exporter` and the `tilecnn` hardware backend remain
  deprecated aliases. Model-zoo releases made before the rename keep their
  `tilecnn_metrics.json` files and `tilecnn` metric keys and are read unchanged.

The reboot marker has no fallback on purpose. A marker left over from an earlier
deployment must never stand in for a new one, so the deploy script deletes the
former marker when it writes the current one.

## What keeps the former name

| Name | Why |
|---|---|
| `vgg16_tilecnn` | A released model ID, recorded in results |
| `data/*_int8_tilecnn/` | Checked-in fixtures written in the former formats; they prove the former identifiers still load |
| Results, papers and presentations dated before the rename | Records are never rewritten, so each stays verifiable against the artifacts that produced it |
