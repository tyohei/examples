#include "common.h"


void init_ranks(proc_info_t *);
bool hostname_exists(char *, void *, int);


void init_MPI(int argc, char **argv, proc_info_t *proc_info) {
  int i = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_info->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_info->size);
  MPI_Get_processor_name(proc_info->hostname, &i);
  init_ranks(proc_info);

  if (0 == proc_info->rank) {
    printf("rank / intra_rank / inter_rank\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);
  for (i=0; i<proc_info->size; i++) {
    if (i == proc_info->rank) {
      printf("%s: %02d / %02d / %02d\n", proc_info->hostname, proc_info->rank,
             proc_info->intra_rank, proc_info->inter_rank);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}


void init_NCCL(proc_info_t proc_info, nccl_info_t *nccl_info) {
  int i = 0;
  int rank = proc_info.rank;
  int size = proc_info.size;
  int root = 0;
  CUDACHECK( cudaSetDevice(proc_info.intra_rank) );

  if (root == rank) {
    NCCLCHECK( ncclGetUniqueId(&nccl_info->nccl_id) );
  }
  MPI_Bcast(&nccl_info->nccl_id, sizeof(nccl_info->nccl_id), MPI_BYTE, root,
            MPI_COMM_WORLD);
  NCCLCHECK( ncclCommInitRank(
        &nccl_info->nccl_comm, size, nccl_info->nccl_id, rank) );
  NCCLCHECK( ncclCommUserRank(nccl_info->nccl_comm, &nccl_info->rank) );
  NCCLCHECK( ncclCommCount(nccl_info->nccl_comm, &nccl_info->size) );
  NCCLCHECK( ncclCommCuDevice(nccl_info->nccl_comm, &nccl_info->device) );

  if (root == rank) {
    printf("rank / device ID\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);
  for (i=0; i<size; i++) {
    if (i == rank) {
      printf("%s: %02d / %02d\n", proc_info.hostname, nccl_info->rank,
             nccl_info->device);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}


void init_buffers(DTYPE **sendbuff_h, DTYPE **recvbuff_h, DTYPE **sendbuff_d,
                  DTYPE **recvbuff_d, int count) {
  *sendbuff_h = (DTYPE *)calloc(sizeof(DTYPE), count);
  *recvbuff_h = (DTYPE *)calloc(sizeof(DTYPE), count);
  CUDACHECK( cudaMalloc((void **)sendbuff_d, sizeof(DTYPE) * count) );
  CUDACHECK( cudaMalloc((void **)recvbuff_d, sizeof(DTYPE) * count) );
}


void free_buffers(DTYPE *sendbuff_h, DTYPE *recvbuff_h, DTYPE *sendbuff_d,
                  DTYPE *recvbuff_d) {
  CUDACHECK( cudaFree(recvbuff_d) );
  CUDACHECK( cudaFree(sendbuff_d) );
  free(recvbuff_h);
  free(sendbuff_h);
}


void init_ranks(proc_info_t *proc_info) {
  int i               = 0;
  int n               = MPI_MAX_PROCESSOR_NAME;
  int rank            = proc_info->rank;
  int size            = proc_info->size;
  char *hostname      = proc_info->hostname;
  void *hostnames     = malloc(sizeof(char) * size * n);
  void *hostnames_set = malloc(sizeof(char) * size * n);
  char *hostname_i    = NULL;
  void *target        = (char *)hostnames + rank * n;

  memcpy((void *)target, (void *)hostname, sizeof(char) * n);

  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostnames, n, MPI_BYTE,
                MPI_COMM_WORLD);

  proc_info->intra_rank = 0;
  proc_info->intra_size = 0;
  proc_info->inter_rank = 0;
  proc_info->inter_size = 0;
  for (i=0; i<size; i++) {
    hostname_i = (char *)hostnames + i * n;
    if (strcmp(hostname_i, hostname) == 0) {
      (proc_info->intra_size)++;
      if (i == rank) {
        proc_info->intra_rank = proc_info->intra_size - 1;
      }
    }

    if (hostname_exists(hostname_i, hostnames_set, proc_info->inter_size)) {
      if (i == rank) {
        proc_info->inter_rank = proc_info->inter_size - 1;
      }
    }
    else {
      target = (char *)hostnames_set + proc_info->inter_size * n;
      memcpy((void *)target, (void *)hostname_i, sizeof(char) * n);
      (proc_info->inter_size)++;
      if (i == rank) {
        proc_info->inter_rank = proc_info->inter_size - 1;
      }
    }
  }
  free(hostnames_set);
  free(hostnames);
}


bool hostname_exists(char *hostname, void *hostnames_set, int length) {
  if (length == 0) {
    return false;
  }
  else {
    int i            = 0;
    int n            = MPI_MAX_PROCESSOR_NAME;
    char *hostname_i = NULL;
    for (i=0; i<length; i++) {
      hostname_i = (char *)hostnames_set + i * n;
      if (strcmp(hostname_i, hostname) == 0) {
        return true;
      }
    }
    return false;
  }
}


void finalize(nccl_info_t nccl_info) {
  ncclCommDestroy(nccl_info.nccl_comm);
  MPI_Finalize();
}


void print_result(struct timeval tvs, struct timeval tve, int count) {
  double sec = (double)(tve.tv_sec - tvs.tv_sec);
  double usec = (double)(tve.tv_usec - tvs.tv_usec) * 0.001 * 0.001;  // micro
  printf("COUNT %d MiB, TIME %lf s\n", count / 104856, sec + usec);
}
