#include "common.h"
#include "reduce.h"
#include "allreduce.h"
#include "allgather.h"


int main(int argc, char **argv)
{
  if (argc < 3)
  {
    fprintf(stderr, "main: %d: invalid number of arguments.\n", __LINE__);
    return 1;
  }
  int i = 0;
  int j = 0;

  proc_info_t proc_info;
  init_MPI(argc, argv, &proc_info);

  nccl_info_t nccl_info;
  init_NCCL(proc_info, &nccl_info);

  DTYPE *sendbuff_h = NULL;
  DTYPE *recvbuff_h = NULL;
  DTYPE *sendbuff_d = NULL;
  DTYPE *recvbuff_d = NULL;
  int counts[11] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 724};
  int length = sizeof(counts) / sizeof(counts[0]);
  int mebi = 104856;  // 2^{20}
  for (i=0; i<length; i++) {
    counts[i] *= mebi;
  }
  init_buffers(&sendbuff_h, &recvbuff_h, &sendbuff_d, &recvbuff_d,
               counts[length - 1]);

  char *communication = argv[1];
  char *library = argv[2];
  int iteration = 10;
  int count = 0;
  /* ======== Reduce ======== */
  if (strcmp("reduce", communication) == 0) {
    void (*reduce_func)(DTYPE *, DTYPE *, DTYPE *, DTYPE *, proc_info_t,
                        nccl_info_t, int) = NULL;
    if (strcmp("nccl", library) == 0) {
      reduce_func = &reduce_NCCL;
      if (proc_info.rank == 0) {
        printf("======== NCCL ========\n");
      }
    }
    else {
      reduce_func = &reduce_MPI;
      if (proc_info.rank == 0) {
        printf("======== MPI ========\n");
      }
    }
    for (j=0; j<length; j++) {
      count = counts[j];
      MPI_Barrier(MPI_COMM_WORLD);
      for (i=0; i<iteration; i++) {
        reduce_func(sendbuff_h, recvbuff_h, sendbuff_d, recvbuff_d, proc_info,
                    nccl_info, count);
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }
  else if (strcmp("allreduce", communication) == 0) {
    void (*allreduce_func)(DTYPE *, DTYPE *, DTYPE *, DTYPE *, proc_info_t,
                           nccl_info_t, int) = NULL;
    if (strcmp("nccl", library) == 0) {
      allreduce_func = &allreduce_NCCL;
      if (proc_info.rank == 0) {
        printf("======== NCCL ========\n");
      }
    }
    else {
      allreduce_func = &allreduce_MPI;
      if (proc_info.rank == 0) {
        printf("======== MPI ========\n");
      }
    }

    for (j=0; j<length; j++) {
      count = counts[j];
      MPI_Barrier(MPI_COMM_WORLD);
      for (i=0; i<iteration; i++) {
        allreduce_func(sendbuff_h, recvbuff_h, sendbuff_d, recvbuff_d,
                       proc_info, nccl_info, count);
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }
  else if (strcmp("allgather", communication) == 0) {
    void (*allgather_func)(DTYPE *, DTYPE *, DTYPE *, DTYPE *, proc_info_t,
                           nccl_info_t, int) = NULL;
    if (strcmp("nccl", library) == 0) {
      allgather_func = &allgather_NCCL;
      if (proc_info.rank == 0) {
        printf("======== NCCL ========\n");
      }
    }
    else {
      allgather_func = &allgather_MPI;
      if (proc_info.rank == 0) {
        printf("======== MPI ========\n");
      }
    }

    for (j=0; j<length; j++) {
      count = counts[j];
      MPI_Barrier(MPI_COMM_WORLD);
      for (i=0; i<iteration; i++) {
        allgather_func(sendbuff_h, recvbuff_h, sendbuff_d, recvbuff_d,
                       proc_info, nccl_info, count);
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }

  if (proc_info.rank == 0) {
    printf("DONE\n");
  }

  free_buffers(sendbuff_h, recvbuff_h, sendbuff_d, recvbuff_d);
  finalize(nccl_info);
}
