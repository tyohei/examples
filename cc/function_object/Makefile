# -------- Files and directories --------
SRC = $(wildcard *.cc)
BIN = $(patsubst %.cc, %, $(SRC))
SUBDIRS :=

# -------- Commands and flags --------
CXX      = g++
CXXFLAGS = -std=c++11 -Wall -O0
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
