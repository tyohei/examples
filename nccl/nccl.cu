#include <cuda.h>
#include <mpi.h>
#include <nccl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


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


/**
 * Check if ``hostname`` exists in ``hostnames_set``.
 * If the ``hostnames_set`` containes ``hostname``, returns 1, otherwise
 * returns 0. ``inter_size`` is the number of current hostname in
 * ``hostnames_set``.
 */
int hostname_exists(
    char *hostname, void *hostnames_set, int *inter_size) {
  if ((*inter_size) == 0) { /* Nothing in the ``hostnames_set``. */
    return 0;
  }
  else {
    char *hostname_i;
    for (int i=0; i<(*inter_size); ++i) {
      hostname_i = (char*)(hostnames_set) + i*MPI_MAX_PROCESSOR_NAME;
      if (strcmp(hostname, hostname_i) == 0) return 1;
    }
    return 0;
  }
}

int init_rank(
    MPI_Comm mpi_comm, void *hostnames, int mpi_rank, int mpi_size,
    int *inter_rank, int *inter_size,
    int *intra_rank, int *intra_size) {
  /**
   * Do the MPI_Allgather() to share hostnames between ranks.
   */
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
      hostnames, MPI_MAX_PROCESSOR_NAME, MPI_BYTE, mpi_comm);

  char *hostname = (char*)hostnames + mpi_rank*MPI_MAX_PROCESSOR_NAME;
  void *hostnames_set = malloc(sizeof(char)*mpi_size*MPI_MAX_PROCESSOR_NAME);
  char *hostname_i;
  (*inter_rank) = 0;
  (*inter_size) = 0;
  (*intra_rank) = 0;
  (*intra_size) = 0;
  for (int i=0; i<mpi_size; ++i) {
    hostname_i = (char*)(hostnames) + i*MPI_MAX_PROCESSOR_NAME;
    /**
     * Calculate intra rank and size.
     */
    if (strcmp(hostname_i, hostname) == 0) {
      (*intra_size)++;
      if (i == mpi_rank) {
        (*intra_rank) = (*intra_size) - 1;
      }
    }

    /**
     * Calculate inter rank and size.
     */
    if (hostname_exists(hostname_i, hostnames_set, inter_size)) {
      if (i == mpi_rank) {
        (*inter_rank) = (*inter_size) - 1;
      }
    }
    else {
      char *hostname_i_target = (char*)hostnames_set +
        (*inter_size)*MPI_MAX_PROCESSOR_NAME;
      memcpy(
          (void*)hostname_i_target,
          (void*)hostname_i,
          MPI_MAX_PROCESSOR_NAME*sizeof(char));
      (*inter_size)++;
      if (i == mpi_rank) {
        (*inter_rank) = (*inter_size) - 1;
      }
    }
  }
  free(hostnames_set);
  return 0;
}


int main(int argc, char **argv) {
  /**
   * Initialize MPI and get the rank and size.
   */
  int mpi_rank;
  int mpi_size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  /**
   * Get the hostname of this process.
   */
  char hostname[MPI_MAX_PROCESSOR_NAME];
  int hostname_len;
  MPI_Get_processor_name(hostname, &hostname_len);

  /**
   * Initialize hostnames array.
   * Subtract this process' hostname to this array.
   */
  void *hostnames = malloc(sizeof(char)*mpi_size*MPI_MAX_PROCESSOR_NAME);
  char *hostname_target = (char*)hostnames + mpi_rank*MPI_MAX_PROCESSOR_NAME;
  memcpy(
      (void*)hostname_target,
      (void*)hostname,
      MPI_MAX_PROCESSOR_NAME*sizeof(char));

  /**
   * Get the inter and intra rank and size.
   */
  int inter_rank;
  int inter_size;
  int intra_rank;
  int intra_size;
  init_rank(
      MPI_COMM_WORLD, hostnames, mpi_rank, mpi_size,
      &inter_rank, &inter_size, &intra_rank, &intra_size);

  printf("[%s:%d]: hostname: %s\n", hostname, mpi_rank, hostname_target);
  printf("[%s:%d]: rank: %d, size: %d\n", hostname, mpi_rank, mpi_rank,
      mpi_size);
  printf("[%s:%d]: inter_rank: %d, inter_size: %d\n", hostname, mpi_rank,
      inter_rank, inter_size);
  printf("[%s:%d]: intra_rank: %d, intra_size: %d\n", hostname, mpi_rank,
      intra_rank, intra_size);

  /**
   * Split the communicator to inter-comm and intra-comm.
   * ``MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm* newcomm);``
   * All processes which pass in the same value for ``color`` are assigned
   * to the same communicator. The ``key`` argument determines the rank within
   * each new communicator. The process which passes in the smallest value
   * will be rank 0, the next smallest will be rank 1, and so on.
   */
  //MPI_Comm inter_comm;
  //MPI_Comm intra_comm;
  //int inter_color = intra_rank;
  //int inter_key = inter_rank;
  //int intra_color = inter_rank;
  //int intra_key = intra_rank;
  //MPI_Comm_split(MPI_COMM_WORLD, inter_color, inter_key, &inter_comm);
  //MPI_Comm_split(MPI_COMM_WORLD, intra_color, intra_key, &intra_comm);

  /**
   * Set GPU device based on intra_rank.
   * Allocate device buffers.
   */
  CUDACHECK( cudaSetDevice(intra_rank) );

  /**
   * Generate NCCL ID and share with all processes using MPI_Bcast().
   */
  ncclUniqueId nccl_id;
  //ncclUniqueId inter_nccl_id;
  //ncclUniqueId intra_nccl_id;
  if (mpi_rank == 0) {
    NCCLCHECK( ncclGetUniqueId(&nccl_id) );
  }
  //if (inter_rank == 0) {
  //  ncclGetUniqueId(&inter_nccl_id);
  //}
  //if (intra_rank == 0) {
  //  ncclGetUniqueId(&intra_nccl_id);
  //}
  MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
  //MPI_Bcast(&inter_nccl_id, sizeof(inter_nccl_id), MPI_BYTE, 0, inter_comm);
  //MPI_Bcast(&intra_nccl_id, sizeof(intra_nccl_id), MPI_BYTE, 0, intra_comm);

  /**
   * Initialize NCCL communicator.
   */
  ncclComm_t nccl_comm;
  //ncclComm_t inter_nccl_comm;
  //ncclComm_t intra_nccl_comm;
  //NCCLCHECK( ncclGroupStart() );
  NCCLCHECK( ncclCommInitRank(&nccl_comm, mpi_size, nccl_id, mpi_rank) );
  //NCCLCHECK( ncclGroupEnd() );
  /* Inter NCCL communicator. */
  //ncclGroupStart();
  //ncclCommInitRank(&inter_nccl_comm, inter_size, inter_nccl_id,
  //    inter_rank);
  //ncclGroupEnd();
  ///* Intra NCCL communicator. */
  //ncclGroupStart();
  //ncclCommInitRank(&intra_nccl_comm, intra_size, intra_nccl_id,
  //    intra_rank);
  //ncclGroupEnd();

  /**
   * Finalize.
   */
  //ncclCommDestroy(inter_nccl_comm);
  //ncclCommDestroy(intra_nccl_comm);
  free(hostnames);
  MPI_Finalize();
  return 0;
}
