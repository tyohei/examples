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

  int n = 10;
  double *send_buf = (double *)malloc(sizeof(double) * n);
  for (int i=0; i<n; ++i)
  {
    if (rank == 0)
    {
      send_buf[i] = i;
    }
    else
    {
      send_buf[i] = 0;
    }
  }

  print_buf(rank, size, hostname, send_buf);

  MPI_Request request;
  MPI_Status status;
  if (rank == 0)
  {
    int dst_rank = 1;
    printf("Sending...\n");
    MPI_Isend(send_buf, n, MPI_DOUBLE, dst_rank, 0, MPI_COMM_WORLD, &request);
    printf("Sending done.\n");
    MPI_Wait(&request, &status);
  }
  else if (rank == 1)
  {
    int src_rank = 0;
    sleep(10);
    printf("Recieving...\n");
    MPI_Recv(send_buf, n, MPI_DOUBLE, src_rank, 0, MPI_COMM_WORLD, &status);
    printf("Recieving done.\n");
  }

  print_buf(rank, size, hostname, send_buf);

  MPI_Finalize();
  return 0;
}

