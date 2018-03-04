#ifndef MY_BCAST_H_
#define MY_BCAST_H_

#include "common.h"

void bcast_h2h(double*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t);
void bcast_h2d(double*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t);
void bcast_d2h(double*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t);
void bcast_d2d(double*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t);

void bcast_init(info_t, double**, double**, size_t);
void bcast_finalize(info_t, double*, double*, size_t);

#endif
