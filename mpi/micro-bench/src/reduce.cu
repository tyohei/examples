#include "reduce.h"
#include "common.h"

void reduce_init_values(DTYPE *, DTYPE *, int);
void reduce_check_result(DTYPE *, DTYPE *, DTYPE *, int, proc_info_t);


void reduce_MPI(DTYPE *sendbuff_h, DTYPE *recvbuff_h, DTYPE *sendbuff_d,
                DTYPE *recvbuff_d, proc_info_t proc_info,
                nccl_info_t nccl_info, int count) {
  struct timeval tvs;
  struct timeval tve;

  reduce_init_values(sendbuff_h, sendbuff_d, count);
  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday(&tvs, NULL);
  MPI_Reduce(sendbuff_d, recvbuff_d, count, MPI_DTYPE, MPI_SUM, 0,
             MPI_COMM_WORLD);
  gettimeofday(&tve, NULL);
  reduce_check_result(sendbuff_h, recvbuff_h, recvbuff_d, count, proc_info);
  print_result(tvs, tve, count);
}


void reduce_NCCL(DTYPE *sendbuff_h, DTYPE *recvbuff_h, DTYPE *sendbuff_d,
                 DTYPE *recvbuff_d, proc_info_t proc_info,
                 nccl_info_t nccl_info, int count) {
  cudaStream_t stream;
  struct timeval tvs;
  struct timeval tve;

  reduce_init_values(sendbuff_h, sendbuff_d, count);
  CUDACHECK( cudaStreamCreate(&stream) );
  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday(&tvs, NULL);
  NCCLCHECK( ncclReduce((void *)sendbuff_d, (void *)recvbuff_d, count,
                         NCCL_DTYPE, ncclSum, 0, nccl_info.nccl_comm,
                         stream) );
  gettimeofday(&tve, NULL);
  reduce_check_result(sendbuff_h, recvbuff_h, recvbuff_d, count, proc_info);
  print_result(tvs, tve, count);
}


void reduce_init_values(DTYPE *sendbuff_h, DTYPE *sendbuff_d, int count) {
  memset((void *)sendbuff_h, 0, sizeof(DTYPE) * count);
  CUDACHECK( cudaMemcpy(sendbuff_d, sendbuff_h, sizeof(DTYPE) * count,
                        cudaMemcpyHostToDevice) );
}


void reduce_check_result(DTYPE *sendbuff_h, DTYPE *recvbuff_h,
                         DTYPE *recvbuff_d, int count, proc_info_t proc_info) {
  CUDACHECK( cudaMemcpy(recvbuff_h, recvbuff_d, sizeof(DTYPE) * count,
                        cudaMemcpyDeviceToHost) );
  int i = 0;
  DTYPE eps = 0.001;

  CUDACHECK( cudaMemcpy(recvbuff_h, recvbuff_d, sizeof(DTYPE) * count,
                        cudaMemcpyDeviceToHost) );
  if (0 == proc_info.rank) {
    for (i=0; i<count; i++) {
      DTYPE send = sendbuff_h[i];
      DTYPE recv = recvbuff_h[i];
      if (!(send * proc_info.size - eps <= recv &&
            send * proc_info.size + eps >= recv)) {
        fprintf(stderr, "value not correct: %f != %f\n", send * proc_info.size,
                recv);
        exit(EXIT_FAILURE);
      }
    }
  }
}
