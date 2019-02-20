#include "common.h"
#include "allreduce.h"

void allreduce_init_values(DTYPE *, DTYPE *, DTYPE *, DTYPE *, int);
void allreduce_check_result(DTYPE *, DTYPE *, DTYPE *, int, proc_info_t);


void allreduce_MPI(DTYPE *sendbuff_h, DTYPE *recvbuff_h, DTYPE *sendbuff_d,
                   DTYPE *recvbuff_d, proc_info_t proc_info,
                   nccl_info_t nccl_info, int count) {
  struct timeval tvs;
  struct timeval tve;
  allreduce_init_values(sendbuff_h, recvbuff_h, sendbuff_d, recvbuff_d, count);
  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday(&tvs, NULL);
  MPI_Allreduce(sendbuff_d, recvbuff_d, count, MPI_DTYPE, MPI_SUM,
                MPI_COMM_WORLD);
  gettimeofday(&tve, NULL);
  allreduce_check_result(sendbuff_h, recvbuff_h, recvbuff_d, count, proc_info);
  print_result(tvs, tve, count);
}


void allreduce_NCCL(DTYPE *sendbuff_h, DTYPE *recvbuff_h, DTYPE *sendbuff_d,
                    DTYPE *recvbuff_d, proc_info_t proc_info,
                    nccl_info_t nccl_info, int count) {
  cudaStream_t stream;
  struct timeval tvs;
  struct timeval tve;

  allreduce_init_values(sendbuff_h, recvbuff_h, sendbuff_d, recvbuff_d, count);
  CUDACHECK( cudaStreamCreate(&stream) );
  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday(&tvs, NULL);
  NCCLCHECK( ncclAllReduce((void *)sendbuff_d, (void *)recvbuff_d, count,
                           NCCL_DTYPE, ncclSum, nccl_info.nccl_comm,
                           stream) );
  gettimeofday(&tve, NULL);
  allreduce_check_result(sendbuff_h, recvbuff_h, recvbuff_d, count, proc_info);
  print_result(tvs, tve, count);
}


void allreduce_init_values(DTYPE *sendbuff_h, DTYPE *recvbuff_h,
                           DTYPE *sendbuff_d, DTYPE *recvbuff_d, int count) {
  int i = 0;
  for (i=0; i<count; i++) {
    sendbuff_h[i] = 0;
  }
  CUDACHECK( cudaMemcpy(sendbuff_d, sendbuff_h, sizeof(DTYPE) * count,
                        cudaMemcpyHostToDevice) );
  memset((void *)recvbuff_h, 0, sizeof(DTYPE) * count);
  CUDACHECK( cudaMemcpy(recvbuff_d, recvbuff_h, sizeof(DTYPE) * count,
                        cudaMemcpyHostToDevice) );
}


void allreduce_check_result(DTYPE *sendbuff_h, DTYPE *recvbuff_h,
                            DTYPE *recvbuff_d, int count,
                            proc_info_t proc_info) {
  CUDACHECK( cudaMemcpy(recvbuff_h, recvbuff_d, sizeof(DTYPE) * count,
                        cudaMemcpyDeviceToHost) );
  int i = 0;
  DTYPE eps = 0.001;

  CUDACHECK( cudaMemcpy(recvbuff_h, recvbuff_d, sizeof(DTYPE) * count,
                        cudaMemcpyDeviceToHost) );
  for (i=0; i<count; i++) {
    DTYPE send = sendbuff_h[i];
    DTYPE recv = recvbuff_h[i];
    if (!(send * proc_info.size - eps <= recv &&
          send * proc_info.size + eps >= recv)) {
      fprintf(stderr, "rank: %0d, i: %d, value not correct: %f != %f\n",
              proc_info.rank, i, send * proc_info.size, recv);
      exit(EXIT_FAILURE);
    }
  }
}
