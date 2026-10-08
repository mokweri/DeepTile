# Port to another board

DeepTile is validated on the AMD ZCU104. Porting it to another FPGA board means
building and packaging a new target for that board while keeping the same
contracts, so that models, the compiler, and run records carry over unchanged.
Prebuilt targets for other boards are not provided.

## What stays the same

| Stays the same | Why |
|---|---|
| ModelPackages | Models are hardware-independent; the same package runs on any compatible target |
| The compiler and host source | They are driven by the architecture parameters, not by a particular board |
| The kernel ABI and descriptor format | Defined once in `src/common/kernel_abi.h` |
| The TargetPackage and RunRecord formats | They identify whatever hardware you build |

## What you provide

- **A Vitis platform** for the board (`.xpfm`). Set it with the `PLATFORM`
  variable in `scripts/hw/Makefile`.
- **A matching Linux image and sysroot**, to cross-compile the host executable
  and boot the board.
- **Boot packaging** for the board, producing the equivalent of the ZCU104's
  SD-card package.
- **A deployment method.** The supplied script assumes a PetaLinux boot
  partition, described below.
- **Enough resources.** DSP use scales with $P_{\mathrm{spatial}}$. The
  on-chip buffers need block RAM and UltraRAM; UltraRAM holds the resident input
  cache.

## Steps

1. **Pick a configuration that fits.** Start from a narrow target such as
   $P_{\mathrm{spatial}} = 2$ and check routed utilization before widening. The
   parameters are described in [Build a hardware target](build-a-target.md).
2. **Validate the kernel in HLS** for the new part before implementation:

    ```bash
    make -C scripts hls-csim DEEPTILE_CASE=best_case_3x3_s1
    ```

3. **Build and package the target** with the board's platform, and a `BOARD`
   name so the target is identified correctly:

    ```bash
    make -C scripts hw-all VITIS_MODE=hw BOARD=<board> TARGET_VERSION=v2.0.0 \
        PLATFORM=/path/to/<board>.xpfm
    ```

    The build emits the same `target.json` contract, recording the board, the
    platform, and the architecture.
4. **Check the package** without the board:

    ```bash
    ./build/tools/target_inspect build/hw/<target>/target.json
    ```

5. **Run a model** through `target.json` and confirm that the output matches
   the packaged integer reference exactly. That comparison is the acceptance
   test for a port.

## Boot images and fabric programming

On the ZCU104, the programmable logic is configured at boot from `BOOT.BIN`,
not by loading the FPGA binary afterwards. The TargetPackage therefore records
the boot image and its digest, and the host refuses to run when the board's boot
image differs or has not yet been activated by a reboot.

If your platform programs the fabric differently, adapt the deployment
accordingly. The `boot` group in `target.json` is optional in the schema; a
target without it is treated as having no verifiable fabric identity, so the
host cannot then detect that a stale fabric is running.

## Deploying with the supplied script

`scripts/hw/deploy_zcu104.sh` works for boards that use the same PetaLinux
boot-partition model. Override its settings; `BOARD` is the board's SSH alias:

```bash
BOARD=<board> \
TARGET_DIR=$PWD/build/hw/<target> \
REMOTE_BOOT_DIR=/run/media/<boot-mount> \
EXPECTED_BOOT_DEVICE=/dev/<boot-device> \
./scripts/hw/deploy_zcu104.sh
```

The script refuses to write unless the remote directory is a writable mount
backed by the expected device, which protects the board's other filesystems.
Boards with a different boot model need their own deployment check.

## Scope

The host and compiler are compiled for one architecture, so each target ships
its own host executable, and the TargetPackage detects a mismatched pair. A
single host that schedules several architectures at run time is not part of the
current design.

The detailed contract is in the [TargetPackage specification](../reference/target-package.md).
