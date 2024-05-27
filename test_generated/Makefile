# Default architecture
ARCH ?= x86

OPT = 

# Supported architectures
VALID_ARCHS := msp430 riscv avr arm x86

# Check if the specified ARCH is valid
ifneq ($(filter $(ARCH), $(VALID_ARCHS)),)
    # Include the architecture-specific Makefile if it exists
    -include $(ARCH).mk
else
    $(error Invalid ARCH specified. Valid options are: $(VALID_ARCHS))
endif

.PHONY: help

all: help

help:
    @echo "Available architectures: $(VALID_ARCHS)"
    @echo "Usage: make ARCH=<architecture> [target]"
    @echo "Available targets: all, clean, dump, size, help, compiler_macros"

clean:

size:

compiler_macros:

dump:
