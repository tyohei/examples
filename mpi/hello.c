#include <stdio.h>
#include <mpi.h>


int main(int argc, char **argv) {
  int rank;
  int size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  char hostname[MPI_MAX_PROCESSOR_NAME];
  int hostname_len;
  MPI_Get_processor_name(hostname, &hostname_len);

  printf("[%d/%d: %s]: Hello World!\n", rank, size, hostname);

  MPI_Finalize();
  return 0;
}
