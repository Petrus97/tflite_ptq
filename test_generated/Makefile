# Shell
SHELL = /bin/sh
#
# Compiler options (https://www.gnu.org/software/make/manual/html_node/Implicit-Variables.html)
#
CC=gcc
CFLAGS=-g -Wall -Werror -Wpedantic --save-temps -fverbose-asm -O0
LDLIBS=-lm
LDFLAGS=

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
BIN = a.out.opt
else
BIN = a.out
endif



CFLAGS += -DDEBUG=1 -Wno-unused-variable

#
# Compile the binary
#
all: prebuild $(BIN)

$(BIN): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -I$(INC_DIR) $(LDLIBS)

$(OBJS): $(OBJ_DIR)/%.o:$(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@ -I$(INC_DIR)


#
# Add the operations to do before start the compilation
#
prebuild:
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(BIN)* $(OBJ_DIR)/*
