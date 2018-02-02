#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank;
  int size;
  char hostname[MPI_MAX_PROCESSOR_NAME];
  int hostname_len;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Get_processor_name(hostname, &hostname_len);
  printf("[%s:%d/%d]: Hello World!\n", hostname, rank, size);
  MPI_Barrier(MPI_COMM_WORLD);
  
  /**
   * Split the communicator to multiple intra communicators.
   *
   *    MPI_Comm_split(
   *      MPI_Comm comm,
   *      int color,
   *      int key,
   *      MPI_Comm* newcomm
   *    );
   *
   * All processes which pass in the same value for ``color`` are assigned
   * to the same communicator.
   * The ``key`` argument determines the rank within the each new
   * communicator. The process which passes in the smallest value will be rank
   * 0, the smallest will be rank 1, and so on.
   */
  MPI_Comm intra_comm;
  int intra_color = rank % 2;
  int intra_key = rank;
  /**
   * ``intra_key`` is defined as ``rank % 2``, this means we are going to make
   * two different type of **INTRA** communicator.
   * 
   *    | rank | intra_comm | newrank |
   *    |------|------------|---------|
   *    | 0    | 0          | 0       |
   *    | 1    | 1          | 0       |
   *    | 2    | 0          | 1       |
   *    | 3    | 1          | 1       |
   *    | 4    | 0          | 2       |
   *    | 5    | 1          | 2       |
   *    |...   | ...        | ...     |
   *
   */
  MPI_Comm_split(MPI_COMM_WORLD, intra_color, intra_key, &intra_comm);

  /**
   * Create a inter communicator from two intra communicators.
   * 
   *    MPI_Intercomm_create(
   *      MPI_Comm local_comm,
   *      int local_leader,
   *      MPI_Comm peer_comm,
   *      int remote_leader,
   *      int tag,
   *      MPI_Comm *newintercomm
   *    )
   *
   * ``local_leader`` and ``remote_leader`` will create a bridge between
   * themselves (by checking the ``tag``). This bridge will make it possible
   * to communicate between two intra communicators.
   * The ``peer_comm`` argument do not have to be a ``remote_comm``. Just
   * specify a pair of ``peer_comm`` and ``remote_leader``, then the local
   * leader will be able to determine where is the remote leader.
   *
   *    Leader in intra 0: rank 0, newrank 0
   *    Leader in intra 1: rank 1, newrank 0
   *
   */
  MPI_Comm inter_comm;
  int tag = 1;
  if (intra_color == 0) {
    MPI_Intercomm_create(intra_comm, 0, MPI_COMM_WORLD, 1, tag, &inter_comm);
  } else {
    MPI_Intercomm_create(intra_comm, 0, MPI_COMM_WORLD, 0, tag, &inter_comm);
  }

  /**
   * Allocate memory
   * Datasize is 262144 * 4 = 1 Mi bytes.
   */
  int datasize = 262144;
  float *buffer = (float *)malloc(sizeof(float) * datasize);
  for (int i=0; i<datasize; i++) {
    buffer[i] = (rank == 0) ? i : 0;
  }

  /**
   * Print
   */
  for (int i=0; i<size; i++) {
    if (i == rank) {
      printf("[%s:%d/%d]: B: (%f, %f, %f, %f, ...)!\n",
             hostname, rank, size, buffer[0], buffer[1], buffer[2], buffer[3]);
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  /**
   * Broadcast the data to inter communicator
   * When the communicator of ``MPI_Bcast`` is an inter communicator, the
   * ``root`` argument must be defined as following rule.
   * Denote the group which the sender belongs, as group A.
   * Denote the group which recive the data from group A, as group B.
   * In group A, the root process (the rank which will actually sends the
   * data to all processes in group B) should define ``root`` as ``MPI_ROOT``.
   * Other ranks in group A should define ``root`` as ``MPI_PROC_NULL``.
   * In group B, all processes should define ``root`` as a value of the actual
   * rank (the rank in group A's communicator) of the root processes in group
   * A.
   */
  int root;
  if (intra_color == 0) {
    root = (rank == 0) ? MPI_ROOT : MPI_PROC_NULL;
  } else {
    root = 0;
  }
  MPI_Bcast(buffer, datasize, MPI_FLOAT, root, inter_comm);

  /**
   * Print
   */
  for (int i=0; i<size; i++) {
    if (i == rank) {
      printf("[%s:%d/%d]: A: (%f, %f, %f, %f, ...)!\n",
             hostname, rank, size, buffer[0], buffer[1], buffer[2], buffer[3]);
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  /**
   * Free
   */
  free(buffer);
  MPI_Comm_free(&inter_comm);
  MPI_Comm_free(&intra_comm);
  MPI_Finalize();
}
