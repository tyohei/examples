#include "common.h"
#include "bcast.h"


/**
 * @brief Call MPI_Bcast() with host to host.
 *
 * @param
 *    buf_d: Head address of device buffer of MPI_Bcast().
 *    count: Number of elements in the broadcasted vector. The datatype would
 *           be double.
 *    dtype: Datatype of MPI_Bcast() MUST be MPI_DOUBLE.
 *    root: Root rank for MPI_Bcast()
 *    comm: MPI communicator of MPI_Bcast()
 *
 */
int bcast_h2h(double *buf_d, int count, MPI_Datatype dtype, int root,
            MPI_Comm comm) {
  double *buf_h = NULL;
  int rc = MPI_SUCCESS;

  buf_h = (double *)malloc(sizeof(double) * count);
  CUDACHECK( cudaMemcpy(buf_h, buf_d, sizeof(double) * count,
                        cudaMemcpyDeviceToHost) );
  rc = MPI_Bcast(buf_h, count, dtype, root, comm);
  CUDACHECK( cudaMemcpy(buf_d, buf_h, sizeof(double) * count,
                        cudaMemcpyHostToDevice) );
  free(buf_h);
  return rc;
}


/**
 * @brief Call MPI_Bcast() with host to device.
 *
 * @param
 *    buf_d: Head address of device buffer of MPI_Bcast().
 *    count: Number of elements in the broadcasted vector. The datatype would
 *           be double.
 *    dtype: Datatype of MPI_Bcast() MUST be MPI_DOUBLE.
 *    root: Root rank for MPI_Bcast()
 *    comm: MPI communicator of MPI_Bcast()
 *
 */
int bcast_h2d(double *buf_d, int count, MPI_Datatype dtype, int root,
            MPI_Comm comm) {
  int rank = 0;

  MPI_Comm_rank(comm, &rank);
  if (rank == root) {
    return bcast_h2h(buf_d, count, dtype, root, comm);
  } else {
    return MPI_Bcast(buf_d, count, dtype, root, comm);
  }
}


/**
 * @brief Call MPI_Bcast() with device to host.
 *
 * @param
 *    buf_d: Head address of device buffer of MPI_Bcast().
 *    count: Number of elements in the broadcasted vector. The datatype would
 *           be double.
 *    dtype: Datatype of MPI_Bcast() MUST be MPI_DOUBLE.
 *    root: Root rank for MPI_Bcast()
 *    comm: MPI communicator of MPI_Bcast()
 *
 */
int bcast_d2h(double *buf_d, int count, MPI_Datatype dtype, int root,
            MPI_Comm comm) {
  int rank = 0;

  MPI_Comm_rank(comm, &rank);
  if (rank != root) {
    return bcast_h2h(buf_d, count, dtype, root, comm);
  } else {
    return MPI_Bcast(buf_d, count, dtype, root, comm);
  }
}


/**
 * @brief Call MPI_Bcast() with device to device.
 *
 * @param
 *    buf_d: Head address of device buffer of MPI_Bcast().
 *    count: Number of elements in the broadcasted vector. The datatype would
 *           be double.
 *    dtype: Datatype of MPI_Bcast() MUST be MPI_DOUBLE.
 *    root: Root rank for MPI_Bcast()
 *    comm: MPI communicator of MPI_Bcast()
 *
 */
int bcast_d2d(double *buf_d, int count, MPI_Datatype dtype, int root,
            MPI_Comm comm) {
  return MPI_Bcast(buf_d, count, dtype, root, comm);
}


/**
 * @brief Initalizations for calling MPI_Bcast()
 *
 * @param
 *    info: Process information.
 *    buf_h: Head address of host buffer of MPI_Bcast().
 *    buf_d: Head address of device buffer of MPI_Bcast().
 *    count: Number of elements in the broadcasted vector. The datatype would
 *           be double.
 *
 */
void bcast_init(const info_t info, double **buf_h, double **buf_d,
                const int count) {
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


/**
 * @brief Finalization for calling MPI_Bcast()
 *
 * @param
 *    info: Process information.
 *    buf_h: Head address of host buffer of MPI_Bcast().
 *    buf_d: Head address of device buffer of MPI_Bcast().
 *    count: Number of elements in the broadcasted vector. The datatype would
 *           be double.
 *
 */
void bcast_finalize(const info_t info, double *buf_h, double *buf_d,
                    const int count) {
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
