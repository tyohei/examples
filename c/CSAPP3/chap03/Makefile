# -------- Files and directories --------
SRC = $(wildcard *.c)
ASB = $(patsubst %.c, %.s, $(SRC))
BIN = $(patsubst %.c, %, $(SRC))
SUBDIRS := 

# -------- Commands and flags --------
CC      = gcc-7
CFLAGS  = -std=c99 -Wall -O1
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
	$(RM) $(BIN) $(ASB)
