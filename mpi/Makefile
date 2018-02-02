MPI_HOME = /usr/local

# -------- Files and directories --------
SRC = $(wildcard *.c)
BIN = $(patsubst %.c, %, $(SRC))
SUBDIRS := inter_comm

# -------- Commands and flags --------
CC      = $(MPI_HOME)/bin/mpicc
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
