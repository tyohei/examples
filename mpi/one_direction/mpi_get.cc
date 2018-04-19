#include <cstdio>
#include <cstdlib>
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

  int n_entries = 8192;

  /**
   * First we create a Window which is a memory space that other processes can
   * write and read.
   */
  MPI_Win win_is_done;  // Window that true or not is allocated.
  MPI_Win win_data;     // Window that data is allocated.
  int is_done = 0;
  float *data = (float *)malloc(sizeof(float) * n_entries);

  /**
   * Allocate memory space for window.
   * If the memory space is already allocated, use MPI_Win_create(), if not
   * use MPI_Win_allocate().
   */
  MPI_Win_create((void *)&is_done, sizeof(int), sizeof(int), MPI_INFO_NULL,
                 MPI_COMM_WORLD, &win_is_done);
  MPI_Win_create((void *)data, sizeof(float) * n_entries, sizeof(float),
                 MPI_INFO_NULL, MPI_COMM_WORLD, &win_data);


  /**
   * To ensure only one process is reading and writing the data, we use
   * MPI_Win_lock() and MPI_Win_unlock()
   */
  int window_owner_rank = 0;
  if (rank == 0) {
    /* Accessing own window. */
    MPI_Win_lock(MPI_LOCK_SHARED, window_owner_rank, 0, win_is_done);
    is_done = 1;
    MPI_Win_unlock(window_owner_rank, win_is_done);
  }
  else if (rank == 1) {
    /* Accessing window owned by rank 0. */
    int window_owner_rank = 0;
    MPI_Win_lock(MPI_LOCK_SHARED, window_owner_rank, 0, win_is_done);
    MPI_Get(&is_done, 1, MPI_INT, window_owner_rank, 0, 1, MPI_INT,
            win_is_done);
    MPI_Win_unlock(window_owner_rank, win_is_done);
  }

  printf("[%d/%d: %s]: %d\n", rank, size, hostname, is_done);

  MPI_Finalize();
  return 0;
}
