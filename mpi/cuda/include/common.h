#ifndef MY_COMMON_H_
#define MY_COMMON_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Cuda failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

typedef struct {
  int rank;
  int size;
  int intra_rank;
  int intra_size;
  int inter_rank;
  int inter_size;
  char hostname[MPI_MAX_PROCESSOR_NAME];
} info_t;

bool hostname_exists(char*, void*, int);
void initialize_info(MPI_Comm, void*, info_t*);

#endif  // MY_COMMON_H_
