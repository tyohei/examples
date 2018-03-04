#include "common.h"
#include "bcast.h"


void bcast_h2h(double *buf_d, size_t count, ncclDataType_t dtype,
               ncclComm_t comm, cudaStream_t stream) {
  double *buf_h = NULL;

  buf_h = (double*)malloc(sizeof(double) * count);
  CUDACHECK( cudaMemcpy(buf_h, buf_d, sizeof(double) * count,
                        cudaMemcpyDeviceToHost) );
  NCCLCHECK( ncclBcast((void*)buf_d, count, dtype, 0, comm, stream) );
  CUDACHECK( cudaMemcpy(buf_d, buf_h, sizeof(double) * count,
                        cudaMemcpyHostToDevice) );
  free(buf_h);
}


void bcast_h2d(double *buf_d, size_t count, ncclDataType_t dtype,
               ncclComm_t comm, cudaStream_t stream) {
  int rank = 0;

  ncclCommUserRank(comm, &rank);
  if (rank == 0) {
    bcast_h2h(buf_d, count, dtype, comm, stream);
  } else {
    bcast_d2d(buf_d, count, dtype, comm, stream);
  }
}


void bcast_d2h(double *buf_d, size_t count, ncclDataType_t dtype,
               ncclComm_t comm, cudaStream_t stream) {
  int rank = 0;

  ncclCommUserRank(comm, &rank);
  if (rank != 0) {
    bcast_h2h(buf_d, count, dtype, comm, stream);
  } else {
    bcast_d2d(buf_d, count, dtype, comm, stream);
  }
}


void bcast_d2d(double *buf_d, size_t count, ncclDataType_t dtype,
               ncclComm_t comm, cudaStream_t stream) {
  NCCLCHECK( ncclBcast((void*)buf_d, count, dtype, 0, comm, stream) );
}


void bcast_init(const info_t info, double **buf_h, double **buf_d,
                const size_t count) {
    /* Allocation and initalization of buffers */
  *buf_h = (double*)malloc(sizeof(double) * count);
  CUDACHECK( cudaMalloc((void**)buf_d, sizeof(double) * count) );
  for (int i=0; i<count; ++i) {
    (*buf_h)[i] = (info.rank == 0) ? i : 0;
  }

  /* Host to device */
  CUDACHECK( cudaMemcpy(*buf_d, *buf_h, sizeof(double) * count,
                        cudaMemcpyHostToDevice) );

  /* Print and barrier */
  for (int i=0; i<info.size; i++) {
    if (info.rank == i) {
      printf("[%d/%d: %s]: B (%.2f, %.2f, %.2f, %.2f, ...)\n",
             info.rank, info.size, info.hostname,
             (*buf_h)[0], (*buf_h)[1], (*buf_h)[2], (*buf_h)[3]);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  if (info.rank == info.size - 1) {
    printf("Starting broadcast...\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);

}


void bcast_finalize(const info_t info, double *buf_h, double *buf_d,
                    const size_t count) {
  /* Device to host */
  CUDACHECK( cudaMemcpy(buf_h, buf_d, sizeof(double) * count,
                        cudaMemcpyDeviceToHost) );

  /* Print and barrier */
  for (int i=0; i<info.size; i++) {
    if (info.rank == i) {
      printf("[%d/%d: %s]: A (%.2f, %.2f, %.2f, %.2f, ...)\n",
             info.rank, info.size, info.hostname,
             buf_h[0], buf_h[1], buf_h[2], buf_h[3]);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  if (info.rank == info.size - 1) {
    printf("Broadcast done.\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  /* Free buffers */
  CUDACHECK( cudaFree(buf_d) );
  free(buf_h);
}
