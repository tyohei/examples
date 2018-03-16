#ifndef MY_ALLGATHER_H_
#define MY_ALLGATHER_H_

#include "common.h"

void allgather_h2h(double*, double*, size_t, ncclDataType_t, ncclComm_t,
                   cudaStream_t);
void allgather_d2d(double*, double*, size_t, ncclDataType_t, ncclComm_t,
                   cudaStream_t);

void allgather_init(info_t, double**, double**, double**, double**, size_t);
void allgather_finalize(info_t, double*, double*, double*, double*,  size_t);

#endif
