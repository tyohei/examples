#include "common.h"
#include "allgather.h"
#include "bcast.h"


void test_bcast(info_t info, size_t count, ncclComm_t comm,
          void (*func_ptr)(double*, size_t, ncclDataType_t, ncclComm_t,
                          cudaStream_t)) {
  /* Buffers */
  double *buf_h = NULL;
  double *buf_d = NULL;
  cudaStream_t stream;
  CUDACHECK( cudaStreamCreate(&stream) );

  /* Broadcast */
  bcast_init(info, &buf_h, &buf_d, count);
  func_ptr(buf_d, count, ncclDouble, comm, stream);
  bcast_finalize(info, buf_h, buf_d, count);
}

void test_allgather(info_t info, size_t count, ncclComm_t comm,
                    void (*func_ptr)(double*, double*, size_t, ncclDataType_t,
                                     ncclComm_t, cudaStream_t)) {
  /* Buffers */
  double *send_buf_h = NULL;
  double *send_buf_d = NULL;
  double *recv_buf_h = NULL;
  double *recv_buf_d = NULL;
  cudaStream_t stream;
  CUDACHECK( cudaStreamCreate(&stream) );

  /* AllGather */
  allgather_init(info, &send_buf_h, &send_buf_d, &recv_buf_h, &recv_buf_d,
                 count);
  func_ptr(send_buf_d, recv_buf_d, count, ncclDouble, comm, stream);
  allgather_finalize(info, send_buf_h, send_buf_d, recv_buf_h, recv_buf_d,
                     count);
}


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
  int ndevices;
  initialize_info(MPI_COMM_WORLD, hostnames, &info);
  cudaGetDeviceCount(&ndevices);

  printf("[%s:%d]: hostname: %s\n", info.hostname, info.rank, hostname_target);
  printf("[%s:%d]: rank: %d, size: %d\n", info.hostname, info.rank, info.rank,
      info.size);
  printf("[%s:%d]: inter_rank: %d, inter_size: %d\n", info.hostname, info.rank,
      info.inter_rank, info.inter_size);
  printf("[%s:%d]: intra_rank: %d, intra_size: %d\n", info.hostname, info.rank,
      info.intra_rank, info.intra_size);
  printf("[%s:%d]: # devices: %d\n", info.hostname, info.rank, ndevices);

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

  void (*bcast_func_ptr)(double*, size_t, ncclDataType_t, ncclComm_t,
                         cudaStream_t);
  void (*allgather_func_ptr)(double*, double*, size_t, ncclDataType_t,
                             ncclComm_t, cudaStream_t);
  size_t count;
  int ctype;
  if (argc < 3) {
    fprintf(stderr, "Not valid arguments.\n");
    return 1;
  } else {
    count = atoi(argv[1]);
    ctype = atoi(argv[2]);
    switch (ctype) {
      case 0:
        bcast_func_ptr = bcast_h2h;
        allgather_func_ptr = allgather_h2h;
        if (info.rank == 0) {
          printf("Using h2h broadcast.\n");
          printf("Using h2h allgather.\n");
        }
        break;
      case 1:
        return 0;
        bcast_func_ptr = bcast_h2d;
        if (info.rank == 0) {
          printf("Using h2d broadcast.\n");
        }
        break;
      case 2:
        return 0;
        bcast_func_ptr = bcast_d2h;
        if (info.rank == 0) {
          printf("Using d2h broadcast.\n");
        }
        break;
      case 3:
        bcast_func_ptr = bcast_d2d;
        allgather_func_ptr = allgather_d2d;
        if (info.rank == 0) {
          printf("Using d2d broadcast.\n");
          printf("Using d2d allgather.\n");
        }
        break;
      default:
        fprintf(stderr, "Not valid arguments.\n");
        return 1;
    }
  }

  /* Actual test */
  test_bcast(info, count, nccl_comm, bcast_func_ptr);
  test_allgather(info, count, nccl_comm, allgather_func_ptr);

  /**
   * Finalize.
   */
  ncclCommDestroy(nccl_comm);
  ncclCommDestroy(inter_nccl_comm);
  ncclCommDestroy(intra_nccl_comm);
  cudaDeviceReset();
  free(hostnames);
  MPI_Finalize();
  return 0;
}
