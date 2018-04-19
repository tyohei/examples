# -------- Files and directories --------
SRC = $(wildcard *.cc)
BIN = $(patsubst %.cc, %, $(SRC))
SUBDIRS :=

# -------- Commands and flags --------
CC      = g++
CFLAGS  = -std=c++11 -Wall -O0
RM      = rm -fr

# -------- Rules --------
.PHONY: all $(SUBDIRS)

all: $(BIN) $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

$(BIN): %: %.cc
	$(CC) $(CFLAGS) -o $@ $<

clean: $(SUBDIRS)
	$(RM) $(BIN)

