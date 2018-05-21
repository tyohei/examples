#include "common.h"
#include "reduce.h"


/**
 * @brief Call MPI_Reduce() with host to host.
 *
 * @param
 *    sendbuf_d: Head address of sending device buffer
 *    recvbuf_d: Head address of receiving device buffer
 *    count: Number of elements in the broadcasted vector. The datatype would
 *           be double.
 *    dtype: Datatype of MPI_Allgatherv() MUST be MPI_DOUBLE.
 *    comm: MPI communicator of MPI_Bcast()
 *
 */
int reduce_h2h(double *sendbuf_d, double *recvbuf_d, int count,
               MPI_Datatype dtype, int root, MPI_Comm comm) {
  return 0;
}


/**
 * @brief Call MPI_Reduce() with host to device.
 *
 * @param
 *    sendbuf_d: Head address of sending device buffer
 *    recvbuf_d: Head address of receiving device buffer
 *    count: Number of elements in the broadcasted vector. The datatype would
 *           be double.
 *    dtype: Datatype of MPI_Allgatherv() MUST be MPI_DOUBLE.
 *    comm: MPI communicator of MPI_Bcast()
 *
 */
int reduce_h2d(double *sendbuf_d, double *recvbuf_d, int count,
               MPI_Datatype dtype, int root, MPI_Comm comm) {
  return 0;
}


/**
 * @brief Call MPI_Reduce() with device to host.
 *
 * @param
 *    sendbuf_d: Head address of sending device buffer
 *    recvbuf_d: Head address of receiving device buffer
 *    count: Number of elements in the broadcasted vector. The datatype would
 *           be double.
 *    dtype: Datatype of MPI_Allgatherv() MUST be MPI_DOUBLE.
 *    comm: MPI communicator of MPI_Bcast()
 *
 */
int reduce_d2h(double *sendbuf_d, double *recvbuf_d, int count,
               MPI_Datatype dtype, int root, MPI_Comm comm) {
  return 0;
}


/**
 * @brief Call MPI_Reduce() with device to device.
 *
 * @param
 *    sendbuf_d: Head address of sending device buffer
 *    recvbuf_d: Head address of receiving device buffer
 *    count: Number of elements in the broadcasted vector. The datatype would
 *           be double.
 *    dtype: Datatype of MPI_Allgatherv() MUST be MPI_DOUBLE.
 *    comm: MPI communicator of MPI_Bcast()
 *
 */
int reduce_d2d(double *sendbuf_d, double *recvbuf_d, int count,
               MPI_Datatype dtype, int root, MPI_Comm comm) {
  return MPI_Reduce(sendbuf_d, recvbuf_d, count, dtype, MPI_SUM, root, comm);
}


/**
 * @brief Initalizations for calling MPI_Reduce()
 *
 * @param
 *    info: Process information.
 *    sendbuf_h: Head address of sending host buffer
 *    recvbuf_h: Head address of receiving host buffer
 *    sendbuf_d: Head address of sending device buffer
 *    recvbuf_d: Head address of receiving device buffer
 *    count: Number of elements in the broadcasted vector. The datatype would
 *           be double.
 *
 */
void reduce_init(const info_t info, double **sendbuf_h, double **recvbuf_h,
                 double **sendbuf_d, double **recvbuf_d, const int count)
{
  /* Allocation and initalization of buffers */
  *sendbuf_h = (double *)calloc(sizeof(double), count);
  *recvbuf_h = (double *)calloc(sizeof(double), count);
  CUDACHECK( cudaMalloc((void **)sendbuf_d, sizeof(double) * count) );
  CUDACHECK( cudaMalloc((void **)recvbuf_d, sizeof(double) * count) );
  for (int i=0; i<count; i++) {
    (*sendbuf_h)[i] = i;
  }

  /* Host to device */
  CUDACHECK( cudaMemcpy(*sendbuf_d, *sendbuf_h, sizeof(double) * count,
                        cudaMemcpyHostToDevice) );

  /* Print and barrier */
  for (int i=0; i<info.size; i++) {
    if (info.rank == i) {
      printf("[%d/%d: %s]: B (%.2f, %.2f, %.2f, %.2f, ...)\n",
             info.rank, info.size, info.hostname,
             (*sendbuf_h)[0], (*sendbuf_h)[1], (*sendbuf_h)[2], (*sendbuf_h)[3]);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  if (info.rank == info.size - 1) {
    printf("Starting MPI_Reduce()...\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);
}


/**
 * @brief Finalization for calling MPI_Bcast()
 *
 * @param
 *    info: Process information.
 *    sendbuf_h: Head address of sending host buffer
 *    recvbuf_h: Head address of receiving host buffer
 *    sendbuf_d: Head address of sending device buffer
 *    recvbuf_d: Head address of receiving device buffer
 *    count: Number of elements in the broadcasted vector. The datatype would
 *           be double.
 *
 */
void reduce_finalize(const info_t info, double *sendbuf_h, double *recvbuf_h,
                     double *sendbuf_d, double *recvbuf_d, const int count)
{
  /* Device to host */
  CUDACHECK( cudaMemcpy(recvbuf_h, recvbuf_d, sizeof(double) * count,
                        cudaMemcpyDeviceToHost) );

  /* Print and barrier */
  for (int i=0; i<info.size; i++) {
    if (info.rank == i) {
      printf("[%d/%d: %s]: A (%.2f, %.2f, %.2f, %.2f, ...)\n",
             info.rank, info.size, info.hostname,
             recvbuf_h[0], recvbuf_h[1], recvbuf_h[2], recvbuf_h[3]);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  if (info.rank == info.size - 1) {
    printf("MPI_Reduce() done.\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  /* Free buffers */
  CUDACHECK( cudaFree(recvbuf_d) );
  CUDACHECK( cudaFree(sendbuf_d) );
  free(recvbuf_h);
  free(sendbuf_h);
}
