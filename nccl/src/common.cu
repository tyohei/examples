#include "common.h"


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
