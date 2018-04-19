#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <mpi.h>


void print_buf(int rank, int size, char *hostname, double *buf)
{
  for (int i=0; i<size; ++i)
  {
    if (rank == i)
    {
      printf("[%d/%d: %s]: ", rank, size, hostname);
      printf("[%lf, %lf, %lf, ...]\n", buf[0], buf[1], buf[2]);
    } 
    MPI_Barrier(MPI_COMM_WORLD);
  }
}


int main(int argc, char **argv)
{
  int rank;
  int size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  char hostname[MPI_MAX_PROCESSOR_NAME];
  int hostname_len;
  MPI_Get_processor_name(hostname, &hostname_len);

  int buffer = 0;
  MPI_Request request;
  MPI_Status status;

  if (rank == 0)
  {
    int count = 1;
    int src_rank = 1;
    printf("Recieving...\n");
    MPI_Irecv(&buffer, count, MPI_INT, src_rank, 0, MPI_COMM_WORLD, &request);
    printf("Recieving done\n");

    int flag;
    MPI_Test(&request, &flag, &status);
    if (flag == 0) {
      printf("Waiting 10 sec...");
      sleep(10);
      printf("up\n");
      MPI_Test(&request, &flag, &status);
      if (flag == 0) {
        printf("Timeup... canceling\n");
        MPI_Cancel(&request);
      }
    }
  }
  MPI_Finalize();
  return 0;
}
