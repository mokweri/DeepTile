# Shared Vitis Make defaults, environment checks, and path helpers.
# Values assigned with `:=` may be overridden on the Make command line.

DEBUG := no

# Enable Vitis port diagnostics with DEBUG=yes.
ifeq ($(DEBUG), yes)
VPP_LDFLAGS += --dk list_ports
endif

ifneq ($(VITIS_MODE), hw)
VPP_FLAGS += -g
endif

# Validate required tool environments.
check-vitis:
ifndef XILINX_VITIS
	$(error XILINX_VITIS variable is not set, please set correctly using "source <Vitis_install_path>/Vitis/<Version>/settings64.sh" and rerun)
endif

check-xrt:
ifndef XILINX_XRT
	$(error XILINX_XRT variable is not set, please set correctly using "source /opt/xilinx/xrt/setup.sh" and rerun)
endif

check-platform:
ifndef PLATFORM
	$(error PLATFORM not set. Please set the PLATFORM properly and rerun. Run "make help" for more details.)
endif

check-devices:
ifndef DEVICE
	$(error DEVICE not set. Please set the DEVICE properly and rerun. Run "make help" for more details.)
endif


# Return a filesystem-safe platform name without the .xpfm suffix.
device2xsa = $(strip $(patsubst %.xpfm, % , $(shell basename $(PLATFORM))))

XSA := 
ifneq ($(PLATFORM), )
XSA := $(call device2xsa, $(PLATFORM))
endif


# Removal commands used by downstream Makefiles.
RM = rm -f
RMDIR = rm -rf

ECHO:= @echo
