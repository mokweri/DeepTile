# Prepare the ZCU104

The ZCU104 needs a bootable SD card once before DeepTile can deploy targets over
SSH. DeepTile's hardware build combines AMD's Zynq UltraScale+ MPSoC common
Linux image with the selected ZCU104 platform, the DeepTile boot image, FPGA
binary, and host application. You do not need to create a custom PetaLinux
project.

## 1. Install AMD's common image

Download `xilinx-zynqmp-common-v2024.2.tar.gz` from the
[Vitis Embedded Platforms downloads](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/embedded-platforms/2024-2.html).
Keep its version aligned with the Vitis toolchain used by DeepTile.

Extract the archive and run its installer. The example destination is
user-writable; replace it with any permanent absolute path:

```bash
DEEPTILE_COMMON_IMAGE_DIR="$PWD/xilinx-common-images/xilinx-zynqmp-common-v2024.2"
mkdir -p "$DEEPTILE_COMMON_IMAGE_DIR"
tar -xf xilinx-zynqmp-common-v2024.2.tar.gz
cd xilinx-zynqmp-common-v2024.2
./sdk.sh -d "$DEEPTILE_COMMON_IMAGE_DIR" -y
```

After installation, the directory supplied as `COMMON_SW_DIR` must contain:

```text
xilinx-zynqmp-common-v2024.2/
|-- Image
|-- rootfs.ext4
`-- linux-files/sysroots/cortexa72-cortexa53-xilinx-linux/
```

AMD describes the archive and sysroot installation in
[Common Image for Vitis Embedded Platforms](https://docs.amd.com/r/2024.2-English/ug1742-vitis-release-notes/Common-Image-for-Vitis-Embedded-Platforms)
and [Installing Embedded Platforms](https://docs.amd.com/r/2024.2-English/ug1701-vitis-accelerated-embedded/Installing-Embedded-Platforms).

## 2. Build the initial SD-card image

Initialize Vitis 2024.2, then build the standard P4 ping-pong target. Override
`COMMON_SW_DIR` because its location is local to your workstation:

```bash
source /opt/Xilinx/Vitis/2024.2/settings64.sh

make -C scripts hw-all \
    VITIS_MODE=hw \
    TARGET_VERSION=v2.0.0 \
    COMMON_SW_DIR=/absolute/path/to/xilinx-zynqmp-common-v2024.2
```

If the ZCU104 platform is not under the default Vitis installation, also pass
its `.xpfm` file as `PLATFORM=/absolute/path/to/xilinx_zcu104_base_202420_1.xpfm`.
The supported 2024.2 base platform is listed in AMD's
[Vitis Embedded Platforms](https://docs.amd.com/r/2024.2-English/ug1742-vitis-release-notes/Vitis-Embedded-Platforms).

The packaged disk image is:

```text
build/hw/zcu104-p4-pp_v2.0.0/sd_card_hw/sd_card.img
```

## 3. Write the image to an SD card

Writing a disk image erases the selected device. Use a graphical image writer,
or identify the SD card carefully with `lsblk`, unmount its mounted partitions,
and write the image to the **whole device**, not to a numbered partition:

```bash
lsblk --paths --output NAME,SIZE,MODEL,MOUNTPOINTS
sudo umount /dev/sdX1 /dev/sdX2
sudo dd \
    if=build/hw/zcu104-p4-pp_v2.0.0/sd_card_hw/sd_card.img \
    of=/dev/sdX bs=4M status=progress conv=fsync
```

!!! danger "Verify `/dev/sdX` before writing"
    Replace `/dev/sdX` with the SD card device reported by `lsblk`. Selecting a
    workstation disk destroys its contents. Partition names and count may
    differ; unmount only partitions that actually appear for the card.

AMD demonstrates the same `sd_card.img` workflow in its
[platform creation tutorial](https://docs.amd.com/r/2024.2-English/Vitis-Tutorials-Vitis-Platform-Creation/Application-1-Run-a-PL-Acceleration-Application).

## 4. Boot and enable SSH

1. Power off the ZCU104, insert the SD card, and set the board's boot mode to
   SD according to the ZCU104 board guide.
2. Connect the serial console and Ethernet, then power on the board.
3. Log in as `petalinux`. On a first boot, PetaLinux prompts you to choose a
   password.
4. Find the board address with `ip address`, and verify that it is reachable
   from the development PC.
5. Add the `zcu104` SSH alias described in [Installation](installation.md),
   then test it with `ssh zcu104`.

The board is now ready for DeepTile. For later builds, deploy the complete
target with `scripts/hw/deploy_zcu104.sh` and reboot; reflashing the SD card is
not part of the normal model or target workflow.
