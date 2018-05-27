#ifndef MY_COMMON_H
#define MY_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Cuda failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("NCCL failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define DTYPE float
#define MPI_DTYPE MPI_FLOAT
#define NCCL_DTYPE ncclFloat32

typedef struct {
  int rank;
  int size;
  int intra_rank;
  int intra_size;
  int inter_rank;
  int inter_size;
  char hostname[MPI_MAX_PROCESSOR_NAME];
} proc_info_t;

typedef struct {
  ncclComm_t nccl_comm;
  ncclUniqueId nccl_id;
  int rank;
  int size;
  int device;
} nccl_info_t;

void init_MPI(int, char **, proc_info_t *);
void init_NCCL(proc_info_t, nccl_info_t *);
void init_buffers(DTYPE **, DTYPE **, DTYPE **, DTYPE **, int);
void free_buffers(DTYPE *, DTYPE *, DTYPE *, DTYPE *);
void finalize(nccl_info_t);
void print_result(struct timeval, struct timeval, int);

#endif  // MY_COMMON_H
