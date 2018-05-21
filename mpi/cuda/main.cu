#include "common.h"
#include "allgatherv.h"
#include "bcast.h"


int bcast_test(int count, info_t info,
               int (*bcast_func)(double*, int, MPI_Datatype, int, MPI_Comm)) {
  /* Allocation and initalization of buffer */
  double *buf_h = NULL;
  double *buf_d = NULL;

  /* Broadcast */
  bcast_init(info, &buf_h, &buf_d, count);
  int rc = bcast_func(buf_d, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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

  /* Allgatherv */
  allgatherv_init(info, &sendbuf_h, &recvbuf_h, &sendbuf_d, &recvbuf_d,
                  &recvcounts, &displs, count);
  int sendcount = recvcounts[info.rank];
  int rc = allgatherv_func(sendbuf_d, sendcount, recvbuf_d, recvcounts, displs,
                           MPI_DOUBLE, MPI_COMM_WORLD);
  allgatherv_finalize(info, sendbuf_h, recvbuf_h, sendbuf_d, recvbuf_d,
                      recvcounts, displs, count);

  return rc;
}


int main(int argc, char** argv) {
  int ctype = 0;
  int mtype = 0;
  int (*bcast_func)(double*, int, MPI_Datatype, int, MPI_Comm) = NULL;
  int (*allgatherv_func)(double*, int, double*, int*, int*, MPI_Datatype,
                         MPI_Comm) = NULL;
  info_t info;
  int hostname_len;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &info.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &info.size);
  MPI_Get_processor_name(info.hostname, &hostname_len);

  /* Conts */
  int mibyte = 104856;  // 2^20
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
        count = counts[i] * mibyte;
        bcast_test(count, info, bcast_func);
        printf("count: %d passed...\n", count);
      }

    } else {
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
        count = counts[i] * mibyte;
        allgatherv_test(count, info, allgatherv_func);
        printf("count: %d passed...\n", count);
      }
    }
  }


  MPI_Finalize();
  cudaDeviceReset();
  return 0;
}
