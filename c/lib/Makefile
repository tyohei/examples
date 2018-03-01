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
SHARED_LIB = $(patsubst %.c, lib%.dylib, $(SHARED_SRC))
else
SHARED_LIB = $(patsubst %.c, lib%.so, $(SHARED_SRC))
endif  # ($(shell uname), Darwin)

SUBDIRS := 

# ======== Commands and flags ========
CC      = gcc
CFLAGS  = -std=c99 -Wall -O0 -I.
LDFLAGS = -L.

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
ifeq ($(shell uname), Darwin)
$(SHARED_LIB): lib%.dylib: %.c
	$(CC) $(CFLAGS) -dynamiclib -fPIC -o $@ $<
else
$(SHARED_LIB): lib%.dylib: %.c
	$(CC) $(CFLAGS) -shared -fPIC -o $@ $<
endif  # ($(shell uname), Darwin)

# -------- Create main binary --------
$(BIN): %: %.c $(STATIC_LIB) $(SHARED_LIB)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -lstatic -lshared

clean: $(SUBDIRS)
	$(RM) $(STATIC_LIB) $(STATIC_OBJ) $(SHARED_LIB)
