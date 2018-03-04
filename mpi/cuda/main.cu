#include "common.h"
#include "bcast.h"


int bcast_test(int count, info_t info,
               int (*func_ptr)(double*, int, MPI_Datatype, int, MPI_Comm)) {
  /* Allocation and initalization of buffer */
  double *buf_h = NULL;
  double *buf_d = NULL;

  /* Broadcast */
  bcast_init(info, &buf_h, &buf_d, count);
  int rc = func_ptr(buf_d, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  bcast_finalize(info, buf_h, buf_d, count);

  return rc;
}


int main(int argc, char** argv) {
  int count = 1;
  int ctype = 0;
  int (*func_ptr)(double*, int, MPI_Datatype, int, MPI_Comm) = NULL;
  info_t info;
  int hostname_len;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &info.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &info.size);
  MPI_Get_processor_name(info.hostname, &hostname_len);

  if (argc < 3) {
    fprintf(stderr, "Not valid arguments.\n");
    return 1;
  } else {
    count = atoi(argv[1]);
    ctype = atoi(argv[2]);
    switch (ctype) {
      case 0:
        func_ptr = &bcast_h2h;
        if (info.rank == 0) {
          printf("Using h2h broadcast.\n");
        }
        break;
      case 1:
        func_ptr = &bcast_h2d;
        if (info.rank == 0) {
          printf("Using h2d broadcast.\n");
        }
        break;
      case 2:
        func_ptr = &bcast_d2h;
        if (info.rank == 0) {
          printf("Using d2h broadcast.\n");
        }
        break;
      case 3:
        func_ptr = &bcast_d2d;
        if (info.rank == 0) {
          printf("Using d2d broadcast.\n");
        }
        break;
      default:
        fprintf(stderr, "Not valid arguments.\n");
        return 1;
    }
  }

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

  bcast_test(count, info, func_ptr);

  MPI_Finalize();
  cudaDeviceReset();
  return 0;
}
