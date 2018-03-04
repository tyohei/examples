include Makefile.config
MPI_HOME  ?= /usr/local
CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= /usr/local/cuda

# -------- Files and directories --------
ifneq ($(wildcard $(CUDA_HOME)/*),)
SRC = $(wildcard src/*.cu)
INC = $(wildcard include/*.h)
OBJ = $(patsubst %.cu, %.o, $(SRC))

MAIN_SRC = main.cu
MAIN_OBJ = $(patsubst %.cu, %.o, $(MAIN_SRC))
MAIN_BIN = $(patsubst %.cu, %, $(MAIN_SRC))
endif
SUBDIRS :=

# -------- Commands and flags --------
NVCC         = $(CUDA_HOME)/bin/nvcc
NVCC_GENCODE ?= -gencode=arch=compute_30,code=sm_30 \
								-gencode=arch=compute_35,code=sm_35 \
								-gencode=arch=compute_50,code=sm_50 \
								-gencode=arch=compute_52,code=sm_52 \
								-gencode=arch=compute_60,code=sm_60 \
								-gencode=arch=compute_61,code=sm_61 \
								-gencode=arch=compute_61,code=compute_61
NVCCFLAGS    = $(NVCC_GENCODE) -O0 \
							 -Xcompiler -Wall, \
							 -I$(MPI_HOME)/include, \
							 -I$(NCCL_HOME)/include, \
							 -I./include
LD           = $(MPI_HOME)/bin/mpicc
LDFLAGS      = -lstdc++ -lcudart -lnccl -L$(CUDA_HOME)/lib64 -L$(NCCL_HOME)/lib
RM      = rm -fr

# -------- Rules --------
.PHONY: all $(SUBDIRS)

all: $(MAIN_BIN) $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

$(MAIN_BIN): $(MAIN_OBJ) $(OBJ)
	$(LD) $(LDFLAGS) -o $@ $^

$(MAIN_OBJ): %.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ 

$(OBJ): %.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ 

clean: $(SUBDIRS)
	$(RM) $(BIN) $(OBJ)
