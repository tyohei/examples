# -------- Files and directories --------
SRC = $(wildcard *.c)
BIN = $(patsubst %.c, %, $(SRC))
SUBDIRS := function_pointer \
	         getenv \
					 struct \
					 lib \

# -------- Commands and flags --------
CC      = gcc
CFLAGS  = -std=c99 -Wall -O0
RM      = rm -fr

# -------- Rules --------
.PHONY: all $(SUBDIRS)

all: $(BIN) $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

$(BIN): %: %.c
	$(CC) $(CFLAGS) -o $@ $<

clean: $(SUBDIRS)
	$(RM) $(BIN)
