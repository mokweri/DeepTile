# Builds the ZCU104 host executable, FPGA binary, and deployable SD-card image.

# Target identity and implementation settings.
VITIS_MODE ?= hw
TARGET_VERSION ?= dev
BOARD ?= zcu104
PSUM_BUFFERING ?= ping-pong
PROFILE ?= 0
include ./utils.mk

# Paths
PRJ_ROOT     := $(abspath ../..)
ABS_PRJ_ROOT := $(realpath $(PRJ_ROOT))
ARCH_P_SPATIAL := $(shell awk '$$2 == "P_SPATIAL" {print $$3; exit}' $(PRJ_ROOT)/src/common/arch_config.h)
PSUM_SUFFIX    := $(if $(filter ping-pong,$(PSUM_BUFFERING)),pp,single)
PROFILE_SUFFIX := $(if $(filter 1,$(PROFILE)),-profiled,)
TARGET_BUILD_NAME ?= $(BOARD)-p$(ARCH_P_SPATIAL)-$(PSUM_SUFFIX)$(PROFILE_SUFFIX)_$(TARGET_VERSION)
BUILD_ROOT    ?= $(PRJ_ROOT)/build/$(VITIS_MODE)
BUILD_DIR     ?= $(BUILD_ROOT)/$(TARGET_BUILD_NAME)
override BUILD_DIR := $(abspath $(BUILD_DIR))
TEMP_DIR     := $(BUILD_DIR)/vpp_temp
VPP_LOG_DIRS := $(BUILD_DIR)/vpp_logs

PACKAGE_OUT  := $(BUILD_DIR)/sd_card_$(VITIS_MODE)
DATA_DIR_SRC := $(ABS_PRJ_ROOT)/data
EMCONFIG_FILE := $(BUILD_DIR)/emconfig.json
XRT_INI := $(BUILD_DIR)/xrt.ini
RUN_SCRIPT := $(BUILD_DIR)/run_app.sh

# Sources and artifacts
PROJECT_NAME := deeptile
HOST_EXE     := $(PROJECT_NAME)
BIN_NAME     := $(PROJECT_NAME).xclbin
TARGET_MANIFEST := $(BUILD_DIR)/target.json
TARGET_BUILD_CAPTURE := $(BUILD_DIR)/target-build-capture.json
TARGET_BUILD_RECORD := $(BUILD_DIR)/target-build.json
DATASET_EXE  := dataset_preprocess

HOST_SRCS_CPP := $(PRJ_ROOT)/src/host/main.cpp \
                 $(PRJ_ROOT)/src/compiler/src/schedule_builder.cpp \
                 $(PRJ_ROOT)/src/compiler/src/cost_model.cpp \
                 $(PRJ_ROOT)/src/compiler/src/descriptor_emitter.cpp \
                 $(PRJ_ROOT)/src/compiler/src/graph_loader.cpp \
                 $(PRJ_ROOT)/src/compiler/src/model_package.cpp \
                 $(PRJ_ROOT)/src/compiler/src/target_loader.cpp \
                 $(PRJ_ROOT)/src/compiler/src/graph_compiler.cpp \
                 $(PRJ_ROOT)/src/host/runtime/runtime.cpp \
                 $(PRJ_ROOT)/src/host/runtime/stem_fold.cpp \
                 $(PRJ_ROOT)/src/host/runtime/run_record.cpp \
                 $(PRJ_ROOT)/src/host/runtime/target_artifact.cpp \
                 $(PRJ_ROOT)/src/host/backends/xrt_backend.cpp \
                 $(PRJ_ROOT)/src/host/backends/dataset_eval.cpp \
                 $(PRJ_ROOT)/src/host/dataset/resize.cpp \
                 $(PRJ_ROOT)/src/host/dataset/preprocess.cpp \
                 $(PRJ_ROOT)/src/host/dataset/dataset.cpp \
                 $(PRJ_ROOT)/src/host/dataset/evaluation.cpp \
                 $(PRJ_ROOT)/src/host/dataset/pipeline.cpp \
                 $(PRJ_ROOT)/src/host/dataset/metrics.cpp \
                 $(PRJ_ROOT)/src/host/dataset/image_decode_opencv.cpp
DATASET_SRCS_CPP := $(PRJ_ROOT)/src/host/tools/dataset_preprocess_main.cpp \
                    $(PRJ_ROOT)/src/host/dataset/resize.cpp \
                    $(PRJ_ROOT)/src/host/dataset/preprocess.cpp \
                    $(PRJ_ROOT)/src/host/dataset/dataset.cpp \
                    $(PRJ_ROOT)/src/host/dataset/pipeline.cpp \
                    $(PRJ_ROOT)/src/host/dataset/image_decode_opencv.cpp


HOST_HDRS := $(wildcard $(PRJ_ROOT)/src/host/*.h) \
             $(wildcard $(PRJ_ROOT)/src/host/*.hpp) \
             $(wildcard $(PRJ_ROOT)/src/host/runtime/*.h) \
             $(wildcard $(PRJ_ROOT)/src/host/backends/*.h) \
             $(wildcard $(PRJ_ROOT)/src/host/reference/*.h) \
             $(wildcard $(PRJ_ROOT)/src/host/dataset/*.h) \
             $(wildcard $(PRJ_ROOT)/src/compiler/include/deeptile/compiler/*.h) \
             $(wildcard $(PRJ_ROOT)/src/compiler/include/deeptile/tile/*.hpp) \
             $(wildcard $(PRJ_ROOT)/src/common/*.h) \
             $(wildcard $(PRJ_ROOT)/src/kernel/*.h)
KERNEL_SRCS  := $(PRJ_ROOT)/src/kernel/tile_conv.cpp \
                $(PRJ_ROOT)/src/kernel/data_orch.cpp \
                $(PRJ_ROOT)/src/kernel/conv_engine.cpp \
                $(PRJ_ROOT)/src/kernel/output_postproc.cpp \
                $(PRJ_ROOT)/src/kernel/dw_stage.cpp
KERNEL_HDRS := $(wildcard $(PRJ_ROOT)/src/kernel/*.h) \
               $(wildcard $(PRJ_ROOT)/src/common/*.h)

KERNEL_NAMES := tile_conv
KERNEL_CU_NAME := tile_conv_1
KERNEL_CLOCK_HZ ?= 200000000
PYTHON ?= python3
# Graph embedded in the generated run_app.sh command.
RUN_GRAPH ?= data/resnet50_int8_tilecnn/graph.json
RUNS ?= 5
WARMUP ?= 2
# Fused pointwise-depthwise lowering is selected by the host compiler and does
# not require rebuilding the FPGA binary.
FUSE_DW ?= 0

# Vitis configuration
KERNEL_CONFIG_FILE = $(ABS_PRJ_ROOT)/scripts/hw/kernel_config.cfg
VIVADO_IMPL_CONFIG = $(ABS_PRJ_ROOT)/scripts/hw/vivado_impl.cfg

# Platform and sysroot
PLATFORM ?= /opt/Xilinx/Vitis/2024.2/base_platforms/xilinx_zcu104_base_202420_1/xilinx_zcu104_base_202420_1.xpfm
# Machine-specific paths can be set in an untracked local.mk instead of on
# every command line.
-include $(ABS_PRJ_ROOT)/scripts/hw/local.mk
COMMON_SW_DIR ?= /opt/Xilinx/platforms/xilinx-zynqmp-common-v2024.2

SYSROOT       := $(COMMON_SW_DIR)/linux-files/sysroots/cortexa72-cortexa53-xilinx-linux
SD_IMAGE_FILE := $(COMMON_SW_DIR)/Image
ROOTFS        := $(COMMON_SW_DIR)/rootfs.ext4

# Tool parallelism
NJOBS = 8

# Toolchain and flags
CXX := $(XILINX_VITIS)/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++
VPP := v++
DEEPTILE_GIT_REVISION := $(shell git -C $(PRJ_ROOT) rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
# Dirtiness is scoped to what actually goes into the artifact. A whole-repo
# check reports dirty for an uncommitted note or paper edit, which says nothing
# about the binary and makes the flag useless by always being set. The scope is
# recorded alongside the flag so a reader knows what "clean" was checked over.
DEEPTILE_SOURCE_SCOPE := src scripts
DEEPTILE_SOURCE_DIRTY := $(if $(strip $(shell git -C $(PRJ_ROOT) status --porcelain -- $(DEEPTILE_SOURCE_SCOPE) 2>/dev/null)),1,0)

# Kernel headers included by the host require the HLS arbitrary-precision types.
ifdef XILINX_VITIS_HLS
	CXXFLAGS += -I$(XILINX_VITIS_HLS)/include
endif
ifdef XILINX_HLS
	CXXFLAGS += -I$(XILINX_HLS)/include
endif
ifdef XILINX_VIVADO
	CXXFLAGS += -I$(XILINX_VIVADO)/include
endif

# Host code is optimized. Image preprocessing and the software reference are
# compute-bound loops where an unoptimized build changes behaviour that matters:
# preprocessing throughput decides how much work must overlap inference.
HOST_OPT ?= -O2

# The cross-compiler needs --sysroot while resolving headers, not only at link
# time.
CXXFLAGS += --sysroot=$(SYSROOT) \
            -I$(XILINX_XRT)/include -I$(PRJ_ROOT)/src/host \
            -I$(PRJ_ROOT)/src/compiler/include \
            -I$(PRJ_ROOT)/src/common -I$(PRJ_ROOT)/src/kernel $(HOST_OPT) -g \
            -Wall \
            -std=c++17 -DXILINX -DCL_TARGET_OPENCL_VERSION=120 \
            -DDEEPTILE_GIT_REVISION='"$(DEEPTILE_GIT_REVISION)"' \
            -DDEEPTILE_SOURCE_DIRTY=$(DEEPTILE_SOURCE_DIRTY) \
            -DDEEPTILE_SOURCE_SCOPE='"$(DEEPTILE_SOURCE_SCOPE)"'
CXXLDFLAGS += -L$(SYSROOT)/usr/lib/ -lOpenCL -lpthread -lrt -lstdc++ \
              -lxrt_coreutil -lopencv_core -lopencv_imgcodecs -ljpeg

# OpenCV supplies only the JPEG decoder; resizing is DeepTile-owned because it
# must reproduce the exporter's Pillow output bit-exactly. Requesting only core
# and imgcodecs keeps imgproc from being linked by accident. Defined after
# SYSROOT so the immediate expansion resolves against the target sysroot.
# libjpeg-turbo (OpenCV's JPEG backend, in the sysroot) lets the dataset path
# decode only the region preprocessing reads, with identical pixels.
OPENCV_CFLAGS   := -I$(SYSROOT)/usr/include/opencv4 -DDEEPTILE_HAVE_LIBJPEG
# The dataset tool never opens the device, so it links neither XRT nor OpenCL.
DATASET_LDFLAGS := -L$(SYSROOT)/usr/lib -lpthread -lstdc++ \
                   -lopencv_core -lopencv_imgcodecs -ljpeg

# FPGA compile and link flags
VPPFLAGS += -t $(VITIS_MODE) --platform $(PLATFORM) --save-temps -g
VPPFLAGS += --temp_dir $(TEMP_DIR) --log_dir $(VPP_LOG_DIRS)
VPPFLAGS += -I$(PRJ_ROOT)/src/kernel -I$(PRJ_ROOT)/src/common
VPP_HLSFLAGS += --hls.jobs $(NJOBS) --hls.clock $(KERNEL_CLOCK_HZ):$(KERNEL_NAMES)
# Partial-sum buffering changes kernel hardware and therefore requires a
# distinct FPGA binary and target identity.
ifneq ($(PSUM_BUFFERING),$(filter $(PSUM_BUFFERING),ping-pong single))
$(error PSUM_BUFFERING must be ping-pong or single)
endif
ifeq ($(PSUM_BUFFERING),single)
VPPFLAGS += -DDEEPTILE_NO_PSUM_PP
endif
VPP_LINKFLAGS += --freqhz $(KERNEL_CLOCK_HZ):$(KERNEL_CU_NAME)
VPP_LINKFLAGS += --vivado.synth.jobs $(NJOBS) --vivado.impl.jobs $(NJOBS)

# PROFILE=1 inserts AXI interface and stall monitors for device tracing.
ifeq ($(PROFILE),1)
VPPFLAGS += --profile.exec all:all:all
VPPFLAGS += --profile.data all:all:all
VPPFLAGS += --profile.stall all:all:all
endif

# Targets
.PHONY: all host dataset fpga manifest package run clean xrt_ini gen_run_app

all: host fpga package ## Build host, xclbin, and SD card image 

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR) $(TEMP_DIR) $(VPP_LOG_DIRS)

host: $(BUILD_DIR)/$(HOST_EXE)

$(BUILD_DIR)/$(HOST_EXE): $(HOST_SRCS_CPP) $(HOST_HDRS) | $(BUILD_DIR)
	@echo "Linking host executable: $@"
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) $(HOST_SRCS_CPP) -o $@ $(CXXLDFLAGS)

# Dataset preprocessing tool. Built separately from the XRT host so a dataset
# run needs no bitstream and no device.
dataset: $(BUILD_DIR)/$(DATASET_EXE)

$(BUILD_DIR)/$(DATASET_EXE): $(DATASET_SRCS_CPP) $(HOST_HDRS) | $(BUILD_DIR)
	@echo "Linking dataset tool: $@"
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) $(DATASET_SRCS_CPP) \
		-o $@ $(DATASET_LDFLAGS)

# Compile the kernel object.
$(TARGET_BUILD_CAPTURE): $(KERNEL_SRCS) $(KERNEL_HDRS) Makefile utils.mk \
                         $(KERNEL_CONFIG_FILE) $(VIVADO_IMPL_CONFIG) \
                         target_build_record.py | $(BUILD_DIR)
	$(PYTHON) target_build_record.py capture \
		--output $@ \
		--repo-root $(ABS_PRJ_ROOT) \
		--arch-config $(PRJ_ROOT)/src/common/arch_config.h \
		--abi-header $(PRJ_ROOT)/src/common/kernel_abi.h \
		--kernel-config $(KERNEL_CONFIG_FILE) \
		--vivado-config $(VIVADO_IMPL_CONFIG) \
		$(foreach path,$(KERNEL_SRCS) $(KERNEL_HDRS) Makefile utils.mk $(KERNEL_CONFIG_FILE) $(VIVADO_IMPL_CONFIG),--input $(path)) \
		--setting board=$(BOARD) \
		--setting platform=$(PLATFORM) \
		--setting vitis_mode=$(VITIS_MODE) \
		--setting kernel_name=$(KERNEL_NAMES) \
		--setting compute_unit=$(KERNEL_CU_NAME) \
		--setting clock_hz=$(KERNEL_CLOCK_HZ) \
		--setting psum_buffering=$(PSUM_BUFFERING) \
		--setting instrumentation=$(if $(filter 1,$(PROFILE)),profile,none)

$(TEMP_DIR)/$(KERNEL_NAMES).xo: $(KERNEL_SRCS) $(KERNEL_HDRS) Makefile \
                                  $(TARGET_BUILD_CAPTURE) | $(BUILD_DIR)
	@echo "Compiling kernel: $(KERNEL_NAMES)"
	cd $(BUILD_DIR) && $(VPP) $(VPPFLAGS) $(VPP_HLSFLAGS) -c -k $(KERNEL_NAMES) -o $@ $(KERNEL_SRCS)

# Link the kernel object against the selected platform.
fpga: $(BUILD_DIR)/$(BIN_NAME)
	$(MAKE) manifest

$(BUILD_DIR)/$(BIN_NAME): $(TEMP_DIR)/$(KERNEL_NAMES).xo Makefile $(KERNEL_CONFIG_FILE) $(VIVADO_IMPL_CONFIG) | $(BUILD_DIR)
	@echo "Linking FPGA binary: $@"
	cd $(BUILD_DIR) && $(VPP) $(VPPFLAGS) $(VPP_LINKFLAGS) -l -o $@ $< \
		$(if $(KERNEL_CONFIG_FILE),--config $(KERNEL_CONFIG_FILE)) \
		$(if $(VIVADO_IMPL_CONFIG),--config $(VIVADO_IMPL_CONFIG))
	$(PYTHON) target_build_record.py finalize \
		--capture $(TARGET_BUILD_CAPTURE) \
		--xclbin $@ \
		--output $(TARGET_BUILD_RECORD)

manifest:
	@test -f $(BUILD_DIR)/$(BIN_NAME) || \
		(echo "XCLBIN not found: $(BUILD_DIR)/$(BIN_NAME)"; exit 2)
	$(PYTHON) generate_target_manifest.py \
		--xclbin $(BUILD_DIR)/$(BIN_NAME) \
		--output $(TARGET_MANIFEST) \
		--repo-root $(ABS_PRJ_ROOT) \
		--arch-config $(PRJ_ROOT)/src/common/arch_config.h \
		--abi-header $(PRJ_ROOT)/src/common/kernel_abi.h \
		--kernel-config $(KERNEL_CONFIG_FILE) \
		--vivado-config $(VIVADO_IMPL_CONFIG) \
		--build-record $(TARGET_BUILD_RECORD) \
		--board $(BOARD) \
		--platform $(PLATFORM) \
		--vitis-mode $(VITIS_MODE) \
		--target-version $(TARGET_VERSION) \
		--kernel-name $(KERNEL_NAMES) \
		--compute-unit $(KERNEL_CU_NAME) \
		--clock-hz $(KERNEL_CLOCK_HZ) \
		--psum-buffering $(PSUM_BUFFERING) \
		--instrumentation $(if $(filter 1,$(PROFILE)),profile,none)

# Emulation configuration
emconfig: $(EMCONFIG_FILE)
$(EMCONFIG_FILE): | $(BUILD_DIR)
	cd $(BUILD_DIR) && emconfigutil --platform $(PLATFORM) --od $(BUILD_DIR)

xrt_ini: | $(BUILD_DIR)
ifeq ($(PROFILE),1)
	@echo "[Debug]" > $(XRT_INI)
	@echo "native_xrt_trace = true" >> $(XRT_INI)
	@echo "device_trace = fine" >> $(XRT_INI)
	@echo "stall_trace = all" >> $(XRT_INI)
	@echo "timeline_trace = true" >> $(XRT_INI)
	@echo "device_counters = 1" >> $(XRT_INI)
	@echo "trace_buffer_size = 256M" >> $(XRT_INI)
	@echo "continuous_trace = true" >> $(XRT_INI)
	@echo "trace_buffer_offload_interval_ms = 100" >> $(XRT_INI)
	@echo "" >> $(XRT_INI)
	@echo "[Emulation]" >> $(XRT_INI)
	@echo "debug_mode = batch" >> $(XRT_INI)
	@echo "waveform_trace = off" >> $(XRT_INI)
else
	@echo "[Debug]" > $(XRT_INI)
	@echo "native_xrt_trace = false" >> $(XRT_INI)
	@echo "device_trace = off" >> $(XRT_INI)
	@echo "stall_trace = off" >> $(XRT_INI)
	@echo "timeline_trace = false" >> $(XRT_INI)
	@echo "" >> $(XRT_INI)
	@echo "[Emulation]" >> $(XRT_INI)
	@echo "debug_mode = batch" >> $(XRT_INI)
	@echo "waveform_trace = off" >> $(XRT_INI)
endif

gen_run_app: | $(BUILD_DIR)
	@echo "Generating run_app.sh..."
	@echo 'export LD_LIBRARY_PATH=/mnt:/tmp:$$LD_LIBRARY_PATH' > $(RUN_SCRIPT)
	@echo 'export XILINX_XRT=/usr' >> $(RUN_SCRIPT)
ifeq ($(FUSE_DW),1)
	@echo 'export DEEPTILE_FUSE_DW=1   # Enable fused pointwise-depthwise lowering.' >> $(RUN_SCRIPT)
endif
ifeq ($(VITIS_MODE),$(filter $(VITIS_MODE),hw_emu sw_emu))
	@echo 'export XCL_EMULATION_MODE=$(VITIS_MODE)' >> $(RUN_SCRIPT)
endif
	@echo './$(HOST_EXE) target.json $(RUN_GRAPH) --runs $(RUNS) --warmup $(WARMUP) --csv xrt_profile.csv' >> $(RUN_SCRIPT)
	@echo 'return_code=$$?' >> $(RUN_SCRIPT)
	@echo 'if [ $$return_code -ne 0 ]; then echo "ERROR: Run Failed"; else echo "INFO: Run Success"; fi' >> $(RUN_SCRIPT)
	@chmod +x $(RUN_SCRIPT)

# SD-card package
package: host fpga gen_run_app xrt_ini $(EMCONFIG_FILE)
	@echo "Packaging SD card image to $(PACKAGE_OUT)/"
	@if test -d "$(PACKAGE_OUT)"; then rm -r -- "$(PACKAGE_OUT)"; fi
	@mkdir -p $(PACKAGE_OUT)
	cd $(BUILD_DIR) && $(VPP) -t $(VITIS_MODE) --platform $(PLATFORM) -p $(BUILD_DIR)/$(BIN_NAME) \
		-o $(PACKAGE_OUT)/$(BIN_NAME) \
		--log_dir $(VPP_LOG_DIRS) --temp_dir $(TEMP_DIR) \
		--package.out_dir $(PACKAGE_OUT) \
		$(if $(filter hw,$(VITIS_MODE)),--package.no_image) \
		--package.rootfs $(ROOTFS) \
		--package.sd_file $(SD_IMAGE_FILE) \
		--package.sd_file $(BUILD_DIR)/$(HOST_EXE) \
		--package.sd_file $(TARGET_MANIFEST) \
		--package.sd_file $(RUN_SCRIPT) \
		--package.sd_file $(XRT_INI) \
		--package.sd_dir $(DATA_DIR_SRC) \
		$(if $(filter $(VITIS_MODE),sw_emu hw_emu),--package.sd_file $(EMCONFIG_FILE))
ifeq ($(VITIS_MODE),hw)
	@echo "Sealing boot image into the packaged TargetPackage manifest"
	$(PYTHON) generate_target_manifest.py \
		--attach-boot $(PACKAGE_OUT)/sd_card/BOOT.BIN \
		--manifest $(PACKAGE_OUT)/sd_card/target.json
	$(PYTHON) generate_target_manifest.py \
		--attach-boot $(PACKAGE_OUT)/sd_card/BOOT.BIN \
		--boot-path $(notdir $(PACKAGE_OUT))/sd_card/BOOT.BIN \
		--manifest $(TARGET_MANIFEST)
	@test ! -e "$(BUILD_DIR)/sdcardfat.img" || { echo "Remove or recover the previous sdcardfat.img before packaging."; exit 1; }
	cd $(BUILD_DIR) && $(XILINX_VITIS)/scripts/vitis/util/mkfsImage.sh \
		-s $(PACKAGE_OUT)/sd_card -o $(PACKAGE_OUT)/sd_card.img -m 1 -e $(ROOTFS)
endif

run: package
ifeq ($(VITIS_MODE),$(filter $(VITIS_MODE),hw_emu sw_emu))
	@echo "Launching $(VITIS_MODE) emulation via QEMU..."
	cd $(PACKAGE_OUT) && ./launch_hw_emu.sh \
		-run-app run_app.sh \
		-no-reboot
else
	@echo "Vitis mode is $(VITIS_MODE). Copy $(PACKAGE_OUT) to the SD card."
endif


clean:
	@test "$(dir $(BUILD_DIR))" = "$(abspath $(BUILD_ROOT))/" || \
		(echo "Refusing to clean outside a single target under BUILD_ROOT"; exit 2)
	rm -r -- "$(BUILD_DIR)"
	@echo "Clean complete."
