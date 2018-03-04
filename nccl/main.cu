#include "common.h"


int main(int argc, char **argv) {
  /**
   * Initialize MPI and get the rank and size.
   */
  info_t info;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &info.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &info.size);

  /**
   * Get the hostname of this process.
   */
  int hostname_len;
  MPI_Get_processor_name(info.hostname, &hostname_len);

  /**
   * Initialize hostnames array.
   * Subtract this process' hostname to this array.
   */
  void *hostnames = malloc(sizeof(char)*info.size*MPI_MAX_PROCESSOR_NAME);
  char *hostname_target = (char*)hostnames + info.rank*MPI_MAX_PROCESSOR_NAME;
  memcpy(
      (void*)hostname_target,
      (void*)info.hostname,
      MPI_MAX_PROCESSOR_NAME*sizeof(char));

  /**
   * Get the inter and intra rank and size.
   */
  initialize_info(MPI_COMM_WORLD, hostnames, &info);

  printf("[%s:%d]: hostname: %s\n", info.hostname, info.rank, hostname_target);
  printf("[%s:%d]: rank: %d, size: %d\n", info.hostname, info.rank, info.rank,
      info.size);
  printf("[%s:%d]: inter_rank: %d, inter_size: %d\n", info.hostname, info.rank,
      info.inter_rank, info.inter_size);
  printf("[%s:%d]: intra_rank: %d, intra_size: %d\n", info.hostname, info.rank,
      info.intra_rank, info.intra_size);

  /**
   * Split the communicator to inter-comm and intra-comm.
   * ``MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm* newcomm);``
   * All processes which pass in the same value for ``color`` are assigned
   * to the same communicator. The ``key`` argument determines the rank within
   * each new communicator. The process which passes in the smallest value
   * will be rank 0, the next smallest will be rank 1, and so on.
   */
  MPI_Comm inter_comm;
  MPI_Comm intra_comm;
  int inter_color = info.intra_rank;
  int inter_key   = info.inter_rank;
  int intra_color = info.inter_rank;
  int intra_key   = info.intra_rank;
  MPI_Comm_split(MPI_COMM_WORLD, inter_color, inter_key, &inter_comm);
  MPI_Comm_split(MPI_COMM_WORLD, intra_color, intra_key, &intra_comm);

  /**
   * Set GPU device based on intra_rank.
   * Allocate device buffers.
   */
  CUDACHECK( cudaSetDevice(info.intra_rank) );

  /**
   * Generate NCCL ID and share with all processes using MPI_Bcast().
   */
  ncclUniqueId nccl_id;
  ncclUniqueId inter_nccl_id;
  ncclUniqueId intra_nccl_id;
  if (info.rank == 0) {
    NCCLCHECK( ncclGetUniqueId(&nccl_id) );
  }
  if (info.inter_rank == 0) {
    ncclGetUniqueId(&inter_nccl_id);
  }
  if (info.intra_rank == 0) {
    ncclGetUniqueId(&intra_nccl_id);
  }
  MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&inter_nccl_id, sizeof(inter_nccl_id), MPI_BYTE, 0, inter_comm);
  MPI_Bcast(&intra_nccl_id, sizeof(intra_nccl_id), MPI_BYTE, 0, intra_comm);

  /**
   * Initialize NCCL communicator.
   */
  ncclComm_t nccl_comm;
  ncclComm_t inter_nccl_comm;
  ncclComm_t intra_nccl_comm;

  NCCLCHECK( ncclCommInitRank(&nccl_comm, info.size, nccl_id, info.rank) );
  NCCLCHECK( ncclCommInitRank(&inter_nccl_comm, info.inter_size, inter_nccl_id,
                              info.inter_rank) );
  NCCLCHECK( ncclCommInitRank(&intra_nccl_comm, info.intra_size, intra_nccl_id,
                              info.intra_rank) );

  /**
   * Finalize.
   */
  ncclCommDestroy(nccl_comm);
  ncclCommDestroy(inter_nccl_comm);
  ncclCommDestroy(intra_nccl_comm);
  free(hostnames);
  MPI_Finalize();
  return 0;
}
