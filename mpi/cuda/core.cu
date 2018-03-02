#include "core.h"


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


int bcast_d2d(double *buf_d, int count, MPI_Datatype dtype, int root,
            MPI_Comm comm) {
  return MPI_Bcast(buf_d, count, dtype, root, comm);
}
