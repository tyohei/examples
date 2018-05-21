#include <sys/time.h>

#include "common.h"
#include "allgatherv.h"
#include "bcast.h"
#include "reduce.h"


int bcast_test(int count, info_t info,
               int (*bcast_func)(double*, int, MPI_Datatype, int, MPI_Comm)) {
  /* Allocation and initalization of buffer */
  double *buf_h = NULL;
  double *buf_d = NULL;
  /* Times */
  struct timeval tvs;
  struct timeval tve;

  /* Broadcast */
  bcast_init(info, &buf_h, &buf_d, count);
  gettimeofday(&tvs, NULL);
  int rc = bcast_func(buf_d, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  gettimeofday(&tve, NULL);
  if (info.rank == 0)
    printf("COUNT %d TIME %ld\n", (count/104856)*8, tve.tv_sec - tvs.tv_sec);
  bcast_finalize(info, buf_h, buf_d, count);

  return rc;
}


int allgatherv_test(int count, info_t info,
                    int (*allgatherv_func)(double*, int, double*, int*, int*,
                                           MPI_Datatype, MPI_Comm))
{
  /* Allocation and initalization of buffers */
  double *sendbuf_h = NULL;
  double *recvbuf_h = NULL;
  double *sendbuf_d = NULL;
  double *recvbuf_d = NULL;
  int *recvcounts = NULL;
  int *displs = NULL;
  /* Times */
  struct timeval tvs;
  struct timeval tve;

  /* Allgatherv */
  allgatherv_init(info, &sendbuf_h, &recvbuf_h, &sendbuf_d, &recvbuf_d,
                  &recvcounts, &displs, count);
  int sendcount = recvcounts[info.rank];
  gettimeofday(&tvs, NULL);
  int rc = allgatherv_func(sendbuf_d, sendcount, recvbuf_d, recvcounts, displs,
                           MPI_DOUBLE, MPI_COMM_WORLD);
  gettimeofday(&tve, NULL);
  if (info.rank == 0)
    printf("COUNT %d TIME %ld\n", (count/104856)*8, tve.tv_sec - tvs.tv_sec);
  allgatherv_finalize(info, sendbuf_h, recvbuf_h, sendbuf_d, recvbuf_d,
                      recvcounts, displs, count);

  return rc;
}


int reduce_test(int count, info_t info,
                int (*reduce_func)(double*, double*, int, MPI_Datatype, int,
                                   MPI_Comm))
{
  /* Allocation and initalization of buffers */
  double *sendbuf_h = NULL;
  double *recvbuf_h = NULL;
  double *sendbuf_d = NULL;
  double *recvbuf_d = NULL;
  /* Times */
  struct timeval tvs;
  struct timeval tve;

  /* Reduce */
  reduce_init(info, &sendbuf_h, &recvbuf_h, &sendbuf_d, &recvbuf_d, count);
  gettimeofday(&tvs, NULL);
  int rc = reduce_func(sendbuf_d, recvbuf_d, count, MPI_DOUBLE, 0,
                       MPI_COMM_WORLD);
  gettimeofday(&tve, NULL);
  if (info.rank == 0)
    printf("COUNT %d TIME %ld\n", (count/104856)*8, tve.tv_sec - tvs.tv_sec);
  reduce_finalize(info, sendbuf_h, recvbuf_h, sendbuf_d, recvbuf_d, count);

  return rc;
}


int main(int argc, char** argv) {
  int ctype = 0;
  int mtype = 0;
  info_t info;
  int hostname_len;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &info.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &info.size);
  MPI_Get_processor_name(info.hostname, &hostname_len);

  /* Conts */
  int mi = 104856;  // 2^20
  int counts[20] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
  int counts_len = 20;
  int count = 0;

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

  initialize_info(MPI_COMM_WORLD, hostnames, &info);
  cudaSetDevice(info.inter_rank);

  if (argc < 3) {
    fprintf(stderr, "Not valid arguments.\n");
    return 1;
  } else {
    ctype = atoi(argv[1]);
    mtype = atoi(argv[2]);

    if (ctype == 0) {
      int (*bcast_func)(double*, int, MPI_Datatype, int, MPI_Comm) = NULL;
      if (info.rank == 0)
        printf("Testing MPI_Bcast()...\n");
      switch (mtype) {
        case 0:
          bcast_func = &bcast_h2h;
          if (info.rank == 0) printf("H2H\n");
          break;
        case 1:
          bcast_func = &bcast_h2d;
          if (info.rank == 0) printf("H2D\n");
          break;
        case 2:
          bcast_func = &bcast_d2h;
          if (info.rank == 0) printf("D2H\n");
          break;
        case 3:
          bcast_func = &bcast_d2d;
          if (info.rank == 0) printf("D2D\n");
          break;
        default:
          fprintf(stderr, "error: invalid arguments.\n");
          return 1;
      }
      /* Test */
      for (int i=0; i<counts_len; ++i) {
        count = counts[i] * mi;
        bcast_test(count, info, bcast_func);
        if (info.rank == 0)
          printf("count: %d MiBytes passed...\n", counts[i]*8);
      }

    } else if (ctype == 1) {
      int (*allgatherv_func)(double*, int, double*, int*, int*, MPI_Datatype,
                         MPI_Comm) = NULL;
      if (info.rank == 0)
        printf("Testing MPI_Allgatherv()...\n");
      switch (mtype) {
        case 0:
          allgatherv_func = &allgatherv_h2h;
          if (info.rank == 0) printf("H2H\n");
          break;
        case 1:
          allgatherv_func = &allgatherv_h2d;
          if (info.rank == 0) printf("H2D\n");
          break;
        case 2:
          allgatherv_func = &allgatherv_d2h;
          if (info.rank == 0) printf("D2H\n");
          break;
        case 3:
          allgatherv_func = &allgatherv_d2d;
          if (info.rank == 0) printf("D2D\n");
          break;
        default:
          fprintf(stderr, "error: invalid arguments.\n");
          return 1;
      }
      /* Test */
      for (int i=0; i<counts_len; ++i) {
        count = counts[i] * mi;
        allgatherv_test(count, info, allgatherv_func);
        if (info.rank == 0)
          printf("count: %d MiBytes passed...\n", counts[i] * 8);
      }
    } else {
      int (*reduce_func)(double*, double*, int, MPI_Datatype, int, MPI_Comm)
        = NULL;
      if (info.rank == 0)
        printf("Testing MPI_Reduce()...\n");
      switch (mtype) {
        case 0:
          reduce_func = &reduce_h2h;
          if (info.rank == 0) printf("H2H\n");
          break;
        case 1:
          reduce_func = &reduce_h2d;
          if (info.rank == 0) printf("H2D\n");
          break;
        case 2:
          reduce_func = &reduce_d2h;
          if (info.rank == 0) printf("D2H\n");
          break;
        case 3:
          reduce_func = &reduce_d2d;
          if (info.rank == 0) printf("D2D\n");
          break;
        default:
          fprintf(stderr, "error: invalid arguments.\n");
          return 1;
      }
      /* Test */
      for (int i=0; i<counts_len; ++i) {
        count = counts[i] * mi;
        reduce_test(count, info, reduce_func);
        if (info.rank == 0)
          printf("count: %d MiBytes passed...\n", counts[i] * 8);
      }
    }
  }


  MPI_Finalize();
  cudaDeviceReset();
  return 0;
}
