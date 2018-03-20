# ======== Files and directories ========
SRC = $(wildcard *.java)
CLS = $(patsubst %.java, %.class, $(SRC))
SUBDIRS := 

# ======== Commands and flags ========
JAVAC   = javac
RM      = rm -fr

# ======== Rules ========
.PHONY: all $(SUBDIRS)

all: $(CLS) $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

$(CLS): %.class: %.java
	$(JAVAC) $(SRC)

clean: $(SUBDIRS)
	$(RM) $(CLS)

