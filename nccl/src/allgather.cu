#include "common.h"
#include "allgather.h"


void allgather_h2h(double *send_buf_d, double *recv_buf_d, size_t count,
                   ncclDataType_t dtype, ncclComm_t comm, cudaStream_t stream)
{
  int nranks;
  double *send_buf_h = NULL;
  double *recv_buf_h = NULL;

  NCCLCHECK( ncclCommCount(comm, &nranks) );
  send_buf_h = (double*)malloc(sizeof(double) * count);
  recv_buf_h = (double*)malloc(sizeof(double) * count * nranks);
  CUDACHECK( cudaMemcpy(send_buf_h, send_buf_d, sizeof(double) * count,
                        cudaMemcpyDeviceToHost) );
  NCCLCHECK( ncclAllGather((void*)send_buf_h, (void*)recv_buf_h, count, dtype,
                            comm, stream) );
  CUDACHECK( cudaMemcpy(recv_buf_d, recv_buf_d,
                        sizeof(double) * count * nranks,
                        cudaMemcpyHostToDevice) );
  free(recv_buf_h);
  free(send_buf_h);
}


void allgather_d2d(double *send_buf_d, double *recv_buf_d, size_t count,
                   ncclDataType_t dtype, ncclComm_t comm, cudaStream_t stream)
{
  NCCLCHECK( ncclAllGather((void*)send_buf_d, (void*)recv_buf_d, count, dtype,
                            comm, stream) );
}


void allgather_init(const info_t info, double **send_buf_h,
                    double **send_buf_d, double **recv_buf_h,
                    double **recv_buf_d, const size_t count)
{
  int nranks = info.size;
  /* Allocation and initalization of buffers */
  *send_buf_h = (double*)malloc(sizeof(double) * count);
  *recv_buf_h = (double*)malloc(sizeof(double) * count * nranks);
  CUDACHECK( cudaMalloc((void**)send_buf_d, sizeof(double) * count) );
  CUDACHECK( cudaMalloc((void**)recv_buf_d, sizeof(double) * count * nranks) );
  for (int i=0; i<count; ++i) {
    (*send_buf_h)[i] = i;
  }
  for (int i=0; i<count * nranks; ++i) {
    (*recv_buf_h)[i] = 0;
  }

  /* Host to device */
  CUDACHECK( cudaMemcpy(*send_buf_d, *send_buf_h,
                        sizeof(double) * count,
                        cudaMemcpyHostToDevice) );
  CUDACHECK( cudaMemcpy(*recv_buf_d, *recv_buf_h,
                        sizeof(double) * count * nranks,
                        cudaMemcpyHostToDevice) );

  /* Print and barrier */
  for (int i=0; i<info.size; i++) {
    if (info.rank == i) {
      printf("[%d/%d: %s]: B (%.2f, %.2f, %.2f, %.2f, ...)\n",
             info.rank, info.size, info.hostname,
             (*send_buf_h)[0], (*send_buf_h)[1], (*send_buf_h)[2],
             (*send_buf_h)[3]);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  if (info.rank == info.size - 1) {
    printf("Starting broadcast...\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);

}


void allgather_finalize(const info_t info, double *send_buf_h,
                        double *send_buf_d, double *recv_buf_h,
                        double *recv_buf_d, const size_t count)
{
  int nranks = info.size;
  /* Device to host */
  CUDACHECK( cudaMemcpy(recv_buf_h, recv_buf_d,
                        sizeof(double) * count * nranks,
                        cudaMemcpyDeviceToHost) );

  /* Print and barrier */
  for (int i=0; i<info.size; i++) {
    if (info.rank == i) {
      printf("[%d/%d: %s]: A (\n", info.rank, info.size, info.hostname);
      for (int j=0; j<info.size; j++) {
        printf("                (%.2f, %.2f, %.2f, %.2f, ...),\n",
               recv_buf_h[0 + j * count], recv_buf_h[1 + j * count],
               recv_buf_h[2 + j * count], recv_buf_h[3 + j * count]);
      }
      printf("               )\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  if (info.rank == info.size - 1) {
    printf("Broadcast done.\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  /* Free buffers */
  CUDACHECK( cudaFree(send_buf_d) );
  CUDACHECK( cudaFree(recv_buf_d) );
  free(send_buf_h);
  free(recv_buf_h);
}
