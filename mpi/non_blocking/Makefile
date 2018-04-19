MPI_HOME = /usr/local

# -------- Files and directories --------
SRC = $(wildcard *.cc)
BIN = $(patsubst %.cc, %, $(SRC))
SUBDIRS := 

# -------- Commands and flags --------
CC       = $(MPI_HOME)/bin/mpicc
CXX      = $(MPI_HOME)/bin/mpicxx
CFLAGS   = -std=c99 -Wall -O1
CXXFLAGS = -std=c++11 -Wall -O1
RM       = rm -fr

# -------- Rules --------
.PHONY: all $(SUBDIRS)

all: $(BIN) $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

$(BIN): %: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $<

clean: $(SUBDIRS)
	$(RM) $(BIN)
