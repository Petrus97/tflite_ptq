# Shell
SHELL = /bin/sh
#
# Compiler options (https://www.gnu.org/software/make/manual/html_node/Implicit-Variables.html)
#
CC=arm-none-eabi-gcc
AS=arm-none-eabi-as
LD=arm-none-eabi-ld
OBJDUMP=arm-none-eabi-objdump
CFLAGS= -Wall -Werror -Wpedantic --save-temps -fverbose-asm -Os -g
LDLIBS=
LDFLAGS=


#
# MCU CONFIGS
#
MCU=cortex-m0
CFLAGS += -nostartfiles -mcpu=$(MCU) -mthumb -ffreestanding

#
# Directories
#
SRC_DIR=./src
INC_DIR=./include
OBJ_DIR=./objs
ARM_DIR=./arm

LDFLAGS+=-T $(ARM_DIR)/linker.ld
#
# Automatic variables for .c and .o files
#
SRCS=$(wildcard $(SRC_DIR)/*.c)
OBJS=$(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

ifdef OPT
SRCS := $(filter-out $(SRC_DIR)/model.c, $(SRCS))
else
SRCS := $(filter-out $(SRC_DIR)/model_opt.c, $(SRCS))
endif


ifdef OPT
BIN = arm$(MCU).opt.elf
else
BIN = arm$(MCU).elf
endif



CFLAGS += -DDEBUG=1 -Wno-unused-variable

#
# Compile the binary
#
all: prebuild assemble $(BIN) dump

$(BIN): $(OBJS) $(OBJ_DIR)/startup.o
	$(CC) $(CFLAGS) -o $@ $^ -I$(INC_DIR) $(LDLIBS)

$(OBJS): $(OBJ_DIR)/%.o:$(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@ -I$(INC_DIR)

assemble:
	$(AS) $(ARM_DIR)/startup.s -mcpu=$(MCU) --warn --fatal-warnings -o $(OBJ_DIR)/startup.o

dump:
	$(OBJDUMP) -D $(BIN) > $(OBJ_DIR)/$(BIN).s

#
# Add the operations to do before start the compilation
#
prebuild:
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(BIN)* $(OBJ_DIR)/* *.out *.map *.s *.i *.o *.elf
