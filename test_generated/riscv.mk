ifdef OPT
BIN = riscv.opt.elf
else
BIN = riscv.elf
endif

include common.mk

CC=riscv32-unknown-elf-gcc
LD=riscv32-unknown-elf-ld
READELF=riscv32-unknown-elf-readelf
OBJDUMP=riscv32-unknown-elf-objdump

LDFLAGS += -Wl,-Map,$(BIN).map
LDFLAGS += -Wl,--gc-sections
LDFLAGS += -Wl,--start-group -lgcc -lc
LDFLAGS += -Wl,--end-group



