# Shell
SHELL = /bin/sh
#
# Compiler options (https://www.gnu.org/software/make/manual/html_node/Implicit-Variables.html)
#
CC=avr-gcc
OBJDUMP=avr-objdump
CFLAGS= -Wall -Werror -Wpedantic --save-temps -fverbose-asm -O3 -g
LDLIBS=
LDFLAGS=-Wl,-Map=output.map -Wl,--gc-sections 
# -Wl,--verbose

#
# MCU CONFIGS
#
MCU=atmega2560
F_CPU=16000000UL

#
# Directories
#
SRC_DIR=./src
INC_DIR=./include
OBJ_DIR=./objs

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
BIN = avr.opt.elf
else
BIN = avr.elf
endif



CFLAGS += -DDEBUG=1 -DAVR=1 -Wno-unused-variable -mmcu=$(MCU) -DF_CPU=$(F_CPU)

#
# Compile the binary
#
all: prebuild $(BIN) size dump

$(BIN): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -I$(INC_DIR) $(LDFLAGS)

$(OBJS): $(OBJ_DIR)/%.o:$(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@ -I$(INC_DIR)


size:
	avr-size -C --mcu=$(MCU) $(BIN)
	avr-readelf -S $(BIN)

dump:
	$(OBJDUMP) -D $(BIN) > $(OBJ_DIR)/$(BIN).s

#
# Add the operations to do before start the compilation
#
prebuild:
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(BIN)* $(OBJ_DIR)/*
