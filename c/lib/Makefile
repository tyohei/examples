# ======== Files and directories ========
SRC = main.c
BIN = $(patsubst %.c, %, $(SRC))

STATIC_SRC = static.c
STATIC_INC = $(patsubst %.c, %.h, $(STATIC_SRC))
STATIC_OBJ = $(patsubst %.c, %.o, $(STATIC_SRC))
STATIC_LIB = $(patsubst %.c, lib%.a, $(STATIC_SRC))

SHARED_SRC = shared.c
SHARED_INC = $(patsubst %.c, %.h, $(SHARED_SRC)) 
ifeq ($(shell uname), Darwin)
  SHARED_EXT = dylib
else
  SHARED_EXT = so
endif
SHARED_LIB = $(patsubst %.c, lib%.$(SHARED_EXT), $(SHARED_SRC))

SUBDIRS := 

# ======== Commands and flags ========
CC      = gcc
CFLAGS  = -std=c99 -Wall -O0 -I.
LDFLAGS = -L.
ifeq ($(shell uname), Darwin)
  SHARED_CFLAGS = -dynamiclib
else
  SHARED_CFLAGS = -shared
endif
SHARED_CFLAGS += -fPIC

AR      = ar r

RM      = rm -fr

# ======== Rules=========
.PHONY: all $(SUBDIRS)

all: $(BIN) $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

# -------- Create static library --------
$(STATIC_OBJ): %.o: %.c
	$(CC) $(CFLAGS) -c $<

$(STATIC_LIB): lib%.a: %.o
	$(AR) $@ $<

# -------- Create dynamic library --------
$(SHARED_LIB): lib%.$(SHARED_EXT): %.c
	$(CC) $(CFLAGS) $(SHARED_CFLAGS) -o $@ $<

# -------- Create main binary --------
$(BIN): %: %.c $(STATIC_LIB) $(SHARED_LIB)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -lstatic -lshared

clean: $(SUBDIRS)
	$(RM) $(STATIC_LIB) $(STATIC_OBJ) $(SHARED_LIB)
