#ifndef MY_ALLGATHER_H
#define MY_ALLGATHER_H

#include "common.h"

void allgather_MPI(
    DTYPE *, DTYPE *, DTYPE *, DTYPE *, proc_info_t, nccl_info_t, int);
void allgather_NCCL(
    DTYPE *, DTYPE *, DTYPE *, DTYPE *, proc_info_t, nccl_info_t, int);

#endif  // MY_ALLGATHER_H
