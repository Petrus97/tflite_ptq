# Shell
SHELL = /bin/sh

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

#
# Compiler options (https://www.gnu.org/software/make/manual/html_node/Implicit-Variables.html)
#
CC=gcc
OBJDUMP=objdump
CFLAGS= -Wall -Werror -Wpedantic -Wno-unused-variable --save-temps -fverbose-asm -g
LDLIBS=
LDFLAGS=

OPT_FLAGS=-O3
CFLAGS += $(OPT_FLAGS)

ifdef OPT
SRCS := $(filter-out $(SRC_DIR)/model.c, $(SRCS))
else
SRCS := $(filter-out $(SRC_DIR)/model_opt.c, $(SRCS))
endif

#
# Compile the binary
#
all: prebuild $(BIN) dump

$(BIN): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -I$(INC_DIR) $(LDFLAGS)

$(OBJS): $(OBJ_DIR)/%.o:$(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@ -I$(INC_DIR)

dump:
	$(OBJDUMP) -D $(BIN) > $(OBJ_DIR)/$(BIN).s

#
# Add the operations to do before start the compilation
#
prebuild:
	mkdir -p $(OBJ_DIR)

compiler_macros:
	$(CC) -dM -E - < /dev/null > $(OBJ_DIR)/compiler_macros.i

clean:
	rm -rf $(BIN)* $(OBJ_DIR)/* *.out *.map *.s *.i *.o *.elf
