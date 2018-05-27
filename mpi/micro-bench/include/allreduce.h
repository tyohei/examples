#ifndef MY_ALLREDUCE_H
#define MY_ALLREDUCE_H

#include "common.h"

void allreduce_MPI(
    DTYPE *, DTYPE *, DTYPE *, DTYPE *, proc_info_t, nccl_info_t, int);
void allreduce_NCCL(
    DTYPE *, DTYPE *, DTYPE *, DTYPE *, proc_info_t, nccl_info_t, int);

#endif  // MY_ALLREDUCE_H
