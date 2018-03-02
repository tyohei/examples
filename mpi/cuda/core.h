#ifndef MY_CORE_H_
#define MY_CORE_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Cuda failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


int bcast_h2h(double*, int, MPI_Datatype, int, MPI_Comm);
int bcast_h2d(double*, int, MPI_Datatype, int, MPI_Comm);
int bcast_d2h(double*, int, MPI_Datatype, int, MPI_Comm);
int bcast_d2d(double*, int, MPI_Datatype, int, MPI_Comm);

#endif  // MY_CORE_H_
