#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Cuda failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


typedef struct {
  int rank;
  int size;
  int intra_rank;
  int intra_size;
  int inter_rank;
  int inter_size;
  char hostname[MPI_MAX_PROCESSOR_NAME];
} info_t;


bool hostname_exists(char*, void*, int);
void initialize_info(MPI_Comm, void*, info_t*);
int bcast(int, info_t);


/**
 * @brief Check if `hostname` exists in ``hostnames_set``.
 *
 * @param
 *    hostname: Target hostname searching in ``hostnames_set``.
 *    hostnames_set: Set of hostnames, the number of hostnames contains in
 *                   this set MUST be equal to ``inter_size``.
 *    inter_size: Number of hostnames in ``hostnames_set``.
 *
 * @return
 *    bool: ``true`` if exists, ``false`` if not.
 *
 */
bool hostname_exists(char *hostname, void *hostnames_set, int inter_size) {
  if (inter_size == 0) { /* Nothing in the ``hostnames_set`` */
    return false;
  } else {
    char *hostname_i;
    for (int i=0; i<inter_size; ++i) {
      hostname_i = (char*)(hostnames_set) + i * MPI_MAX_PROCESSOR_NAME;
      if (strcmp(hostname, hostname_i) == 0) {
        return true;
      }
    }
    return false;
  }
}


/**
 * @brief Initialize the MPI processes information.
 *
 * @param
 *    comm: Target MPI communicator for this function.
 *    hostnames: All hostnames in the communicator ``comm``.
 *    info: Information about this process. ``rank``, ``size``, and
 *            ``hostname`` should be specified before this function is called.
 *
 */
void initialize_info(MPI_Comm comm, void *hostnames, info_t *info) {
  // Do the all gather to share hostnames between processes.
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostnames,
      MPI_MAX_PROCESSOR_NAME, MPI_BYTE, comm);

  char *hostname = (char*)hostnames + info->rank * MPI_MAX_PROCESSOR_NAME;
  void *hostnames_set = malloc(sizeof(char) * info->size
      * MPI_MAX_PROCESSOR_NAME);
  char *hostname_i;
  info->intra_rank = 0;
  info->intra_size = 0;
  info->inter_rank = 0;
  info->inter_size = 0;
  for (int i=0; i<info->size; ++i) {
    hostname_i = (char*)hostnames + i * MPI_MAX_PROCESSOR_NAME;
    /* Calculate the intra rank and size */
    if (strcmp(hostname_i, hostname) == 0) {
      (info->intra_size)++;
      if (i == info->rank) {
        info->intra_rank = info->intra_size - 1;
      }
    }

    /* Calculate the inter rank and size */
    if (hostname_exists(hostname_i, hostnames_set, info->inter_size)) {
      if (i == info->rank) {
        info->inter_rank = info->inter_size - 1;
      }
    } else {
      char *hostname_i_target = (char*)hostnames_set + info->inter_size *
        MPI_MAX_PROCESSOR_NAME;
      memcpy((void*)hostname_i_target, (void*)hostname_i,
          MPI_MAX_PROCESSOR_NAME * sizeof(char));
      (info->inter_size)++;
      if (i == info->rank) {
        info->inter_rank = info->inter_size - 1;
      }
    }
  }
  free(hostnames_set);
  return;
}


/**
 * @brief Broadcast a data from rank 0 to others.
 *
 * @param
 *    count: Number of elements in the broadcasted vector. The datatype would
 *           be double.
 *    info: Information about this MPI process.
 *
 * @return
 *    int: Error code of ``MPI_Bcast``.
 *
 */
int bcast(int count, info_t info) {
  cudaSetDevice(info.inter_rank);

  /* Allocation and initalization of buffer */
  double *buf_h = (double*)malloc(sizeof(double) * count);
  double *buf_d = NULL;
  CUDACHECK( cudaMalloc(&buf_d, sizeof(double) * count) );
  for (int i=0; i<count; ++i) {
    buf_h[i] = (info.rank == 0) ? i : 0;
  }

  /* Host to device */
  CUDACHECK( cudaMemcpy(buf_d, buf_h, sizeof(double) * count,
        cudaMemcpyHostToDevice) );

  if (info.rank == 1) {
    printf("[%d/%d: %s]: B (%.2f, %.2f, %.2f, %.2f, ...)\n",
        info.rank, info.size, info.hostname,
        buf_h[0], buf_h[1], buf_h[2], buf_h[3]);
    printf("Starting broadcast...\n");
  }

  /* Broadcast */
  int rc = MPI_Bcast(buf_d, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  /* Device to host */
  CUDACHECK( cudaMemcpy(buf_h, buf_d, sizeof(double) * count,
        cudaMemcpyDeviceToHost) );

  if (info.rank == 1) {
    printf("[%d/%d: %s]: A (%.2f, %.2f, %.2f, %.2f, ...)\n",
        info.rank, info.size, info.hostname,
        buf_h[0], buf_h[1], buf_h[2], buf_h[3]);
    printf("Starting broadcast...\n");
  }

  CUDACHECK( cudaFree(buf_d) );
  free(buf_h);
  cudaDeviceReset();
  return rc;
}


int main(int argc, char** argv) {
  int count = 8192;
  info_t info;
  int hostname_len;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &info.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &info.size);
  MPI_Get_processor_name(info.hostname, &hostname_len);

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
  bcast(count, info);

  MPI_Finalize();
  return 0;
}
