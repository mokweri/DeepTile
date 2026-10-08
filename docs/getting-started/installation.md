# Installation

DeepTile spans three environments: a development PC for building, a Python
environment for model export, and a ZCU104 board for running. What you need
depends on how far you want to go.

| To | You need |
|---|---|
| Build the compiler and host tools, and run the native tests | A C++17 compiler and CMake |
| Export models with FixQuant | A Python environment with PyTorch |
| Simulate the kernel in HLS | Vitis HLS 2024.2 |
| Build hardware targets and deploy them | Vitis/Vivado 2024.2, the ZCU104 platform, and a ZCU104 board |
| Build or preview this documentation | Python 3, Doxygen, and Graphviz |

## Get the source

```bash
git clone --recursive https://github.com/mokweri/DeepTile.git
cd DeepTile
```

FixQuant is a Git submodule at `FixQuant/`. If you cloned without
`--recursive`:

```bash
git submodule update --init --recursive
```

## Development PC

### Native tools

| Requirement | Notes |
|---|---|
| CMake and a C++17 compiler | Build the compiler, host libraries, inspection tools, and tests |
| OpenCV (`core`, `imgcodecs`) | Optional. Needed only to decode JPEG images for dataset evaluation; everything else builds without it |
| libjpeg or libjpeg-turbo | Optional. Lets dataset evaluation decode only the region of each JPEG that the crop needs; without it OpenCV decodes the whole image, with identical results |

Verify the native build:

```bash
make -C scripts tools
ctest --test-dir build/tools --output-on-failure
```

### AMD tools for HLS and hardware

| Requirement | Default location |
|---|---|
| Vitis and Vivado 2024.2 | `/opt/Xilinx/Vitis/2024.2`. Provides `v++`, `vitis_hls`, and the AArch64 cross-compiler |
| ZCU104 base platform | `xilinx_zcu104_base_202420_1` under the Vitis installation |
| PetaLinux common sysroot | `/opt/Xilinx/platforms/xilinx-zynqmp-common-v2024.2`. Linux files for cross-compiling the board host |
| Disk space | About 20 GB for three hardware targets; each build takes 5–6 GB |

Initialize the Vitis environment in each shell before HLS or hardware
commands. The platform and common-image paths are the `PLATFORM` and
`COMMON_SW_DIR` variables in `scripts/hw/Makefile`. If your installation
differs, set them in an untracked `scripts/hw/local.mk`:

```make
COMMON_SW_DIR = /path/to/xilinx-zynqmp-common-v2024.2
```

## Python environment for FixQuant

Model export runs under FixQuant, which needs Python 3.9 or newer. Use any
isolated environment:

```bash
python3 -m venv .venv-fixquant
source .venv-fixquant/bin/activate
pip install torch torchvision numpy pyyaml onnx tqdm Pillow
pip install --no-deps -e ./FixQuant
```

A CPU-only PyTorch build is enough: export does not need a GPU. `--no-deps`
installs FixQuant without letting pip replace the PyTorch build you chose, so
install its dependencies explicitly as shown.

Point DeepTile's build at this interpreter, and check it:

```bash
export FIXQUANT_PYTHON="$(command -v python)"
$FIXQUANT_PYTHON -c "import fixquant, torch; print(torch.__version__)"
```

!!! warning "Set `FIXQUANT_PYTHON`"
    The build defaults to the system `python3`, which cannot import FixQuant.
    Without the export above, model export fails inside the exporter with an
    import error.

## ZCU104 board

DeepTile targets the AMD ZCU104 evaluation board. Before the first deployment,
follow [Prepare the ZCU104](board-setup.md) to install AMD's common image, build
and flash the initial SD-card image, and boot the board.

The running board needs SSH access from the development PC. Add an alias to
`~/.ssh/config`:

```sshconfig
Host zcu104
    HostName <board-IP-address>
    User petalinux
```

Test it with `ssh zcu104`. The deployment script uses this alias, `ssh`, and
`tar`; it does not need `rsync` on the board.

## Documentation

Install Doxygen and Graphviz using your operating system's package manager.
The existing documentation target creates an isolated Python environment and
installs MkDocs and its plugins:

```bash
make -C scripts docs-venv     # once
make -C scripts docs-serve    # then open http://127.0.0.1:8000
```

## Next steps

Continue with the [Quick start](quick-start.md) to export a model, build a
target, and run it on the board.
