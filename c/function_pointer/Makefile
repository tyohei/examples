# -------- Files and directories --------
SRC = $(wildcard *.c)
BIN = $(patsubst %.c, %, $(SRC))
ASB = $(patsubst %.c, %.s, $(SRC))
SUBDIRS := 

# -------- Commands and flags --------
CC      = gcc
CFLAGS  = -std=c99 -Wall -O0
RM      = rm -fr

# -------- Rules --------
.PHONY: all $(SUBDIRS)

all: $(BIN) $(ASB) $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

$(BIN): %: %.c
	$(CC) $(CFLAGS) -o $@ $<

$(ASB): %.s: %.c
	$(CC) $(CFLAGS) -S $<

clean: $(SUBDIRS)
	$(RM) $(BIN)
