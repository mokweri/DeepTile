# Entry point for native tools, HLS validation, and Vitis hardware builds.
#
# Native tools are generated in build/tools, HLS projects in build/hls, and
# hardware packages in build/<VITIS_MODE>.

REPO_ROOT := ..
TOOLS_BUILD_DIR := $(REPO_ROOT)/build/tools
HLS_TCL := hls/run_hls.tcl
HLS_LOG := $(abspath $(REPO_ROOT)/build/hls)/$(or $(DEEPTILE_HLS_COMPONENT),deeptile_hls_proj).log
HW_DIR := hw

CMAKE ?= cmake
CMAKE_CXX ?= /usr/bin/g++
VITIS_HLS ?= vitis_hls
FIXQUANT_PYTHON ?= python3

VITIS_MODE ?= hw
TARGET_VERSION ?= dev
BOARD ?= zcu104
PSUM_BUFFERING ?= ping-pong
PROFILE ?= 0
NJOBS ?= 8
BUILD_DIR ?=
# The former TILECNN_ prefix still works on the command line and in the
# environment: each former name seeds its unset DEEPTILE_ replacement.
FORMER_VARIABLES := HLS_EXEC CASE GRAPH TRACE_LEVEL RESET_PROJECT HLS_COMPONENT \
                    DISABLE_K1_CACHE DISABLE_CIG_CHAIN NO_PSUM_PP
$(foreach v,$(FORMER_VARIABLES),$(if $(TILECNN_$(v)),$(eval DEEPTILE_$(v) ?= $(TILECNN_$(v)))))
DEEPTILE_HLS_EXEC ?= 3
DEEPTILE_CASE ?= best_case_3x3_s1
DEEPTILE_TRACE_LEVEL ?= none
DEEPTILE_RESET_PROJECT ?= 1
FUSE_DW ?= 0
ZOO_MODEL ?=
# Documentation toolchain, installed privately under build/ by docs-venv.
DOCS_VENV ?= $(abspath $(REPO_ROOT)/build/docs-venv)
MKDOCS := $(DOCS_VENV)/bin/mkdocs
DOXYGEN ?= doxygen

HW_ARGS := VITIS_MODE=$(VITIS_MODE) TARGET_VERSION=$(TARGET_VERSION) \
	BOARD=$(BOARD) PSUM_BUFFERING=$(PSUM_BUFFERING) PROFILE=$(PROFILE) \
	NJOBS=$(NJOBS)
ifneq ($(strip $(BUILD_DIR)),)
HW_ARGS += BUILD_DIR=$(BUILD_DIR)
endif

.PHONY: help tools graph_plan graph_plan_tool runtime_tests reference_tests model-fetch model-export hls hls-csim hls-csynth hls-cosim hls-export hls-cosim-reuse hw-host hw-dataset hw-fpga hw-manifest hw-package hw-all hw-run hw-clean clean-tools docs-venv docs-api docs docs-serve docs-clean

help:
	@echo "DeepTile build orchestration"
	@echo ""
	@echo "Native tools:"
	@echo "  make tools                         Configure/build native graph/runtime tools"
	@echo "  make graph_plan                    Build only graph_plan"
	@echo "  make runtime_tests                 Build and run runtime_tests"
	@echo "  make reference_tests               Check all data/ packages against exported refs"
	@echo ""
	@echo "Model packages:"
	@echo "  make model-fetch ZOO_MODEL=<release-id>"
	@echo "  make model-export ZOO_MODEL=<release-id>"
	@echo ""
	@echo "HLS:"
	@echo "  make hls-csim DEEPTILE_CASE=<case>"
	@echo "  make hls-csynth DEEPTILE_CASE=<case>"
	@echo "  make hls-cosim DEEPTILE_CASE=<case> DEEPTILE_TRACE_LEVEL=none|port|all"
	@echo "  make hls-export"
	@echo "  make hls DEEPTILE_HLS_EXEC=<1..6> DEEPTILE_CASE=<case>"
	@echo ""
	@echo "Hardware:"
	@echo "  make hw-host VITIS_MODE=hw"
	@echo "  make hw-dataset VITIS_MODE=hw"
	@echo "  make hw-fpga VITIS_MODE=hw NJOBS=8"
	@echo "  make hw-manifest VITIS_MODE=hw"
	@echo "  make hw-package VITIS_MODE=hw TARGET_VERSION=v2.0.0"
	@echo "  make hw-all VITIS_MODE=hw"
	@echo "  make hw-run VITIS_MODE=hw_emu"
	@echo "  make hw-clean VITIS_MODE=hw"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs-venv                     Install MkDocs into build/docs-venv"
	@echo "  make docs-serve                    Build the API reference and preview the site"
	@echo "  make docs                          Build the static site into build/docs/site"
	@echo "  make docs-clean                    Remove generated documentation"

tools:
	$(CMAKE) -S . -B $(TOOLS_BUILD_DIR) -DCMAKE_CXX_COMPILER=$(CMAKE_CXX)
	$(CMAKE) --build $(TOOLS_BUILD_DIR) -j

graph_plan:
	$(CMAKE) -S . -B $(TOOLS_BUILD_DIR) -DCMAKE_CXX_COMPILER=$(CMAKE_CXX)
	$(CMAKE) --build $(TOOLS_BUILD_DIR) --target graph_plan -j

# Alias retained for existing automation.
graph_plan_tool: graph_plan

runtime_tests:
	$(CMAKE) -S . -B $(TOOLS_BUILD_DIR) -DCMAKE_CXX_COMPILER=$(CMAKE_CXX)
	$(CMAKE) --build $(TOOLS_BUILD_DIR) --target host_tests run_record_tests -j
	$(TOOLS_BUILD_DIR)/host_tests
	$(TOOLS_BUILD_DIR)/run_record_tests

# Reconciles the software reference against every exported reference in data/.
reference_tests:
	$(CMAKE) -S . -B $(TOOLS_BUILD_DIR) -DCMAKE_CXX_COMPILER=$(CMAKE_CXX)
	$(CMAKE) --build $(TOOLS_BUILD_DIR) --target reference_equivalence_tests -j
	$(TOOLS_BUILD_DIR)/reference_equivalence_tests

model-fetch:
	@test -n "$(ZOO_MODEL)" || \
		(echo "ZOO_MODEL is required: model/dataset/profile@version"; exit 2)
	cd $(REPO_ROOT)/FixQuant && scripts/model_zoo.sh fetch "$(ZOO_MODEL)"

model-export: graph_plan
	@test -n "$(ZOO_MODEL)" || \
		(echo "ZOO_MODEL is required: model/dataset/profile@version"; exit 2)
	cd $(REPO_ROOT) && $(FIXQUANT_PYTHON) scripts/export_model.py \
		--zoo-model "$(ZOO_MODEL)"

hls:
	@mkdir -p $(dir $(HLS_LOG))
	DEEPTILE_HLS_EXEC=$(DEEPTILE_HLS_EXEC) \
	DEEPTILE_CASE=$(DEEPTILE_CASE) \
	DEEPTILE_TRACE_LEVEL=$(DEEPTILE_TRACE_LEVEL) \
	DEEPTILE_RESET_PROJECT=$(DEEPTILE_RESET_PROJECT) \
	DEEPTILE_DISABLE_K1_CACHE=$(DEEPTILE_DISABLE_K1_CACHE) \
	DEEPTILE_DISABLE_CIG_CHAIN=$(DEEPTILE_DISABLE_CIG_CHAIN) \
	DEEPTILE_NO_PSUM_PP=$(DEEPTILE_NO_PSUM_PP) \
	$(VITIS_HLS) -f $(HLS_TCL) -l $(HLS_LOG)

hls-csim:
	$(MAKE) hls DEEPTILE_HLS_EXEC=1

hls-csynth:
	$(MAKE) hls DEEPTILE_HLS_EXEC=2

hls-cosim:
	$(MAKE) hls DEEPTILE_HLS_EXEC=3

hls-export:
	$(MAKE) hls DEEPTILE_HLS_EXEC=4

hls-cosim-reuse:
	$(MAKE) hls DEEPTILE_HLS_EXEC=6 DEEPTILE_RESET_PROJECT=0

hw-host:
	$(MAKE) -C $(HW_DIR) host $(HW_ARGS)

hw-dataset:
	$(MAKE) -C $(HW_DIR) dataset $(HW_ARGS)

hw-fpga:
	$(MAKE) -C $(HW_DIR) fpga $(HW_ARGS)

hw-manifest:
	$(MAKE) -C $(HW_DIR) manifest $(HW_ARGS)

hw-package:
	$(MAKE) -C $(HW_DIR) package $(HW_ARGS) FUSE_DW=$(FUSE_DW)

hw-all:
	$(MAKE) -C $(HW_DIR) all $(HW_ARGS)

hw-run:
	$(MAKE) -C $(HW_DIR) run $(HW_ARGS)

hw-clean:
	$(MAKE) -C $(HW_DIR) clean $(HW_ARGS)

clean-tools:
	rm -rf $(TOOLS_BUILD_DIR)

docs-venv:
	python3 -m venv $(DOCS_VENV)
	$(DOCS_VENV)/bin/pip install -q -r $(REPO_ROOT)/docs/requirements.txt

# The C++ API reference is generated into docs/reference/api/ so that MkDocs
# serves it as static files, both in the live preview and in the built site.
docs-api:
	mkdir -p $(REPO_ROOT)/build/docs
	rm -rf $(REPO_ROOT)/docs/reference/api
	cd $(REPO_ROOT) && $(DOXYGEN) docs/doxygen/Doxyfile

docs: docs-api
	cd $(REPO_ROOT) && $(MKDOCS) build --strict

docs-serve: docs-api
	cd $(REPO_ROOT) && $(MKDOCS) serve

docs-clean:
	rm -rf $(REPO_ROOT)/build/docs $(REPO_ROOT)/docs/reference/api
