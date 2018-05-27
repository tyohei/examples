#ifndef MY_REDUCE_H
#define MY_REDUCE_H

#include "common.h"

void reduce_MPI(
    DTYPE *, DTYPE *, DTYPE *, DTYPE *, proc_info_t, nccl_info_t, int);
void reduce_NCCL(
    DTYPE *, DTYPE *, DTYPE *, DTYPE *, proc_info_t, nccl_info_t, int);

#endif  // MY_REDUCE_H
