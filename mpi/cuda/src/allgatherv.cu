#include "common.h"
#include "allgatherv.h"


/**
 * @brief Call MPI_Allgatherv() with host to host.
 *
 * @param
 *    sendbuf_d: Head address of sending device buffer of MPI_Allgatherv()
 *    sendcount: Number of elements in the sendbuf_d.
 *    recvbuf_d: Head address of receiving device buffer of MPI_Allgatherv()
 *    recvcounts: Array of number of elements that are to be received from
 *      each process
 *    displs: Entry i specifies the displacement at which to place the
 *      incoming data from process i.
 *    dtype: Datatype of MPI_Allgatherv() MUST be MPI_DOUBLE.
 *    comm: MPI communicator of MPI_Bcast()
 *
 */
int allgatherv_h2h(double *sendbuf_d, int sendcount, double *recvbuf_d,
                   int *recvcounts, int *displs, MPI_Datatype dtype,
                   MPI_Comm comm) {
  double *sendbuf_h = NULL;
  double *recvbuf_h = NULL;
  int rc = MPI_SUCCESS;
  int rank = 0;
  int size = 1;
  int total_recvcount = 0;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  for (int i=0; i<size; i++)
    total_recvcount += recvcounts[i];

  sendbuf_h = (double *)malloc(sizeof(double) * sendcount);
  recvbuf_h = (double *)malloc(sizeof(double) * total_recvcount);

  CUDACHECK( cudaMemcpy(sendbuf_h, sendbuf_d, sizeof(double) * sendcount,
                        cudaMemcpyDeviceToHost) );
  rc = MPI_Allgatherv(sendbuf_h, sendcount, dtype, recvbuf_h, recvcounts,
                      displs, dtype, comm);
  CUDACHECK( cudaMemcpy(recvbuf_d, recvbuf_h, sizeof(double) * total_recvcount,
                        cudaMemcpyHostToDevice) );
  free(recvbuf_h);
  free(sendbuf_h);
  return rc;
}


/**
 * @brief Call MPI_Allgatherv() with host to device.
 *
 * @param
 *    sendbuf_d: Head address of sending device buffer of MPI_Allgatherv()
 *    sendcount: Number of elements in the sendbuf_d.
 *    recvbuf_d: Head address of receiving device buffer of MPI_Allgatherv()
 *    recvcounts: Array of number of elements that are to be received from
 *      each process
 *    displs: Entry i specifies the displacement at which to place the
 *      incoming data from process i.
 *    dtype: Datatype of MPI_Allgatherv() MUST be MPI_DOUBLE.
 *    comm: MPI communicator of MPI_Bcast()
 *
 */
int allgatherv_h2d(double *sendbuf_d, int sendcount, double *recvbuf_d,
                   int *recvcounts, int *displs, MPI_Datatype dtype,
                   MPI_Comm comm) {
  /* NOT IMPLEMENTED ! */
  return 0;
}


/**
 * @brief Call MPI_Allgatherv() with device to host.
 *
 * @param
 *    sendbuf_d: Head address of sending device buffer of MPI_Allgatherv()
 *    sendcount: Number of elements in the sendbuf_d.
 *    recvbuf_d: Head address of receiving device buffer of MPI_Allgatherv()
 *    recvcounts: Array of number of elements that are to be received from
 *      each process
 *    displs: Entry i specifies the displacement at which to place the
 *      incoming data from process i.
 *    dtype: Datatype of MPI_Allgatherv() MUST be MPI_DOUBLE.
 *    comm: MPI communicator of MPI_Bcast()
 *
 */
int allgatherv_d2h(double *sendbuf_d, int sendcount, double *recvbuf_d,
                   int *recvcounts, int *displs, MPI_Datatype dtype,
                   MPI_Comm comm) {
  /* NOT IMPLEMENTED ! */
  return 0;
}


/**
 * @brief Call MPI_Allgatherv() with device to host.
 *
 * @param
 *    sendbuf_d: Head address of sending device buffer of MPI_Allgatherv()
 *    sendcount: Number of elements in the sendbuf_d.
 *    recvbuf_d: Head address of receiving device buffer of MPI_Allgatherv()
 *    recvcounts: Array of number of elements that are to be received from
 *      each process
 *    displs: Entry i specifies the displacement at which to place the
 *      incoming data from process i.
 *    dtype: Datatype of MPI_Allgatherv() MUST be MPI_DOUBLE.
 *    comm: MPI communicator of MPI_Bcast()
 *
 */
int allgatherv_d2d(double *sendbuf_d, int sendcount, double *recvbuf_d,
                   int *recvcounts, int *displs, MPI_Datatype dtype,
                   MPI_Comm comm) {
  return MPI_Allgatherv(sendbuf_d, sendcount, dtype, recvbuf_d, recvcounts,
                        displs, dtype, comm);
}


/**
 * @brief Initalizations for calling MPI_Allgatherv()
 *
 * @param
 *    info: Process information.
 *    sendbuf_h: Head address of host buffer (output)
 *    recvbuf_h: Head address of device buffer (output)
 *    sendbuf_d: Head address of host buffer (output)
 *    recvbuf_d: Head address of device buffer (output)
 *    recvcounts: Head address of recvcounts array (output)
 *    displs: Head of displs array (output)
 *    count: Number of elements in the broadcasted vector. The datatype would
 *           be double.
 *
 */
void allgatherv_init(const info_t info, double **sendbuf_h, double **recvbuf_h,
                     double **sendbuf_d, double **recvbuf_d, int **recvcounts,
                     int **displs, const int count) {
  /* Calculate the sendcount and recvcount */
  int recvcounts_sum = 0;
  *recvcounts = (int *)malloc(sizeof(int) * info.size);
  *displs = (int *)malloc(sizeof(int) * info.size);
  for (int i=0; i<info.size; i++) {
    (*displs)[i] = recvcounts_sum;
    (*recvcounts)[i] = count / info.size;
    recvcounts_sum += (*recvcounts)[i];
  }
  (*recvcounts)[info.size - 1] = count - recvcounts_sum;

  /* Allocation and initalization of buffers */
  *sendbuf_h = (double *)calloc((*recvcounts)[info.rank], sizeof(double));
  *recvbuf_h = (double *)calloc(count, sizeof(double));
  for (int i=0; i<(*recvcounts)[info.rank]; i++) {
    (*sendbuf_h)[i] = i;
  }

  CUDACHECK( cudaMalloc((void**)sendbuf_d,
                        sizeof(double) * (*recvcounts)[info.rank]) );
  CUDACHECK( cudaMalloc((void**)recvbuf_d,
                        sizeof(double) * count) );

  /* Host to device */
  CUDACHECK( cudaMemcpy(*sendbuf_d, *sendbuf_h,
                        sizeof(double) * (*recvcounts)[info.rank],
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
    printf("Starting broadcast...\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);
}


/**
 * @brief Finalization for calling MPI_Allgatherv()
 *
 * @param
 *    info: Process information.
 *    sendbuf_h: Head address of host buffer (output)
 *    recvbuf_h: Head address of device buffer (output)
 *    sendbuf_d: Head address of host buffer (output)
 *    recvbuf_d: Head address of device buffer (output)
 *    recvcounts: Head address of recvcounts array (output)
 *    displs: Head of displs array (output)
 *    count: Number of elements in the broadcasted vector. The datatype would
 *           be double.
 *
 */
void allgatherv_finalize(const info_t info, double *sendbuf_h,
                         double *recvbuf_h, double *sendbuf_d,
                         double *recvbuf_d, int *recvcounts, int *displs,
                         const int count) {
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
    printf("Broadcast done.\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  /* Free buffers */
  CUDACHECK( cudaFree(recvbuf_d) );
  CUDACHECK( cudaFree(sendbuf_d) );
  free(displs);
  free(recvcounts);
  free(recvbuf_h);
  free(sendbuf_h);
}
