# ======== Files and directories ========
SUFFIX := $(shell python-config --extension-suffix)
SHARED_SRC = $(wildcard *.cc)
SHARED_OBJ = $(patsubst %.cc, %.o, $(SHARED_SRC))
SHARED_LIB = $(patsubst %.cc, %$(SUFFIX), $(SHARED_SRC))

SUBDIRS := 


# ======== Commands and flags ========
INC = $(shell python -m pybind11 --includes)

CXX      = g++
CXXFLAGS = -std=c++11 -Wall -O3 $(INC)
RM       = rm -fr

SHARED_CXXFLAGS = $(CXXFLAGS) 
ifeq ($(shell uname), Darwin)
  SHARED_CXXFLAGS += -dynamiclib -undefined dynamic_lookup
else
  SHARED_CXXFLAGS += -shared
endif
SHARED_CXXFLAGS += -fPIC


# ======== Rules ========
.PHONY: all clean

all: $(SHARED_LIB) $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

$(SHARED_LIB): %$(SUFFIX): %.o
	$(CXX) $(SHARED_CXXFLAGS) -o $@ $^

$(SHARED_OBJ): %.o: %.cc
	$(CXX) $(CXXFLAGS) -c $<

clean: $(SUBDIRS)
	$(RM) $(SHARED_OBJ) $(SHARED_LIB)
