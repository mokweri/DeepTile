# Run a model on the board

This guide deploys a hardware target to the ZCU104 and runs a model on it. It
assumes you have a built target under `build/hw/` and an exported ModelPackage
under `build/models/` — see the [Quick start](../getting-started/quick-start.md)
if not.

## 1. Deploy the target

Package the target and deploy it:

```bash
D=build/hw/zcu104-p4-pp_v2.0.0
make -C scripts/hw package VITIS_MODE=hw TARGET_VERSION=v2.0.0 \
    BUILD_DIR=$PWD/$D -o $PWD/$D/deeptile.xclbin -o $PWD/$D/deeptile
TARGET_DIR=$PWD/$D ./scripts/hw/deploy_zcu104.sh
```

The script verifies that it is writing to the board's real boot partition, then
streams the files over SSH. When the target contains an SD-card package it
performs a full deployment, including `BOOT.BIN`; otherwise it updates only the
host executable, FPGA binary, and `target.json`. It never touches the board's
`data/` and `models/` directories.

## 2. Reboot to activate the fabric

```bash
ssh zcu104 'sudo reboot'
```

!!! warning "A full deployment is not active until you reboot"
    On the ZCU104 the programmable logic is configured at boot from `BOOT.BIN`.
    Copying a new FPGA binary cannot change the running hardware. Until you
    reboot, the board still runs the previous fabric, and the host refuses to
    execute because the new boot image has not been activated.

The SSH connection closing during reboot is expected. Give the board time to
boot and obtain its network address. To confirm that the right hardware is
running, compare the boot image's digest with the one in the target manifest:

```bash
ssh zcu104 'sha256sum ~/boot/BOOT.BIN'   # must equal boot.sha256 in target.json
```

A host-only update does not need a reboot when its boot image is already
active.

## 3. Copy the model to the board

Models are deployed separately from targets, so one target can run many
networks:

```bash
MODEL_REL=resnet50/imagenet1k/int8-tqt/v1.0.0
tar czf - -C build/models "$MODEL_REL" |
    ssh zcu104 'mkdir -p ~/boot/models && cd ~/boot/models && tar xzf -'
```

Later target deployments preserve this directory.

## 4. Run it

Connect to the board and change to the boot partition:

```bash
ssh zcu104
cd /run/media/boot-mmcblk0p1          # also reachable as ~/boot
export LD_LIBRARY_PATH=/mnt:/tmp:${LD_LIBRARY_PATH:-}
export XILINX_XRT=/usr
```

!!! note "`~/boot` is not `/boot`"
    On this PetaLinux image `~/boot` links to the boot partition. `/boot` is an
    unrelated directory on the root filesystem.

Run the model's packaged validation input and check it against the exported
reference:

```bash
./deeptile target.json models/resnet50/imagenet1k/int8-tqt/v1.0.0/graph.json \
    --runs 5 --warmup 2 --output results/resnet50-single
```

The host compiles the graph for the target, verifies the FPGA binary and boot
image, runs the input, and compares the output with the packaged integer
reference. Any mismatch is reported as an error.

### Running depthwise-separable networks

For MobileNetV2 and EfficientNet-Lite0, enable fused producer → depthwise
execution:

```bash
DEEPTILE_FUSE_DW=1 \
./deeptile target.json models/mobilenet_v2/imagenet1k/int8-tqt-cle/v1.0.0/graph.json \
    --runs 5 --warmup 2 --output results/mobilenet_v2-single
```

!!! warning "Omitting `DEEPTILE_FUSE_DW` fails silently"
    Without the variable, the network still runs correctly and reaches the same
    accuracy, but MobileNetV2 lowers to 53 launches instead of 36 and
    EfficientNet-Lite0 to 50 instead of 34, and both run more slowly. Check
    `launches` and `fused_depthwise` in `run.json` before quoting a performance
    figure.

## Useful options

| Option | Effect |
|---|---|
| `--runs N` | Measured runs (default 1) |
| `--warmup N` | Warm-up runs before measuring (default 1) |
| `--output DIR` | Write `run.json` and a per-launch profile; refuses to overwrite an existing record |
| `--no-compare` | Skip the reference comparison |
| `--measure-latency` | Time input upload through output download for each run |
| `--verbose` | Print per-launch detail |

The full list is in the [command reference](../reference/commands.md#host-executable).
For accuracy over a dataset, see [Evaluate on a dataset](evaluate-on-a-dataset.md).

## Tips

- **Discard the first run after each boot.** It carries a large one-time cost.
- **Save records before rebooting** if you wrote them under `/tmp`, which is
  cleared at boot.
- **Upload missing test fixtures by hand.** The deploy script never copies
  fixtures:

    ```bash
    tar czf - -C data <suite> | ssh zcu104 'cd /run/media/boot-mmcblk0p1/data && tar xzf -'
    ```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| The host reports that the boot image is pending activation | You deployed a full package without rebooting. Reboot the board |
| The host reports a boot-image or FPGA-binary checksum mismatch | The board holds a different target. Deploy the full package for this target and reboot |
| Output does not match the reference | Check that the host executable came from the same target build as the FPGA binary |
