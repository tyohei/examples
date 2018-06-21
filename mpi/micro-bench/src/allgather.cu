#include "common.h"
#include "allgather.h"

void allgather_init_values(DTYPE *, DTYPE *, DTYPE *, DTYPE *, int);
void allgather_check_result(DTYPE *, DTYPE *, DTYPE *, int, proc_info_t);


void allgather_MPI(DTYPE *sendbuff_h, DTYPE *recvbuff_h, DTYPE *sendbuff_d,
                   DTYPE *recvbuff_d, proc_info_t proc_info,
                   nccl_info_t nccl_info, int count) {
  int sendcount = count / proc_info.size;
  struct timeval tvs;
  struct timeval tve;
  allgather_init_values(sendbuff_h, recvbuff_h, sendbuff_d, recvbuff_d, count);
  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday(&tvs, NULL);
  MPI_Allgather(sendbuff_d + sendcount * proc_info.rank, sendcount, MPI_DTYPE,
                recvbuff_d, count, MPI_DTYPE, MPI_COMM_WORLD);
  gettimeofday(&tve, NULL);
  allgather_check_result(sendbuff_h, recvbuff_h, recvbuff_d, count, proc_info);
  print_result(tvs, tve, count);
}


void allgather_NCCL(DTYPE *sendbuff_h, DTYPE *recvbuff_h, DTYPE *sendbuff_d,
                    DTYPE *recvbuff_d, proc_info_t proc_info,
                    nccl_info_t nccl_info, int count) {
  cudaStream_t stream;
  int sendcount = count / proc_info.size;
  struct timeval tvs;
  struct timeval tve;

  allgather_init_values(sendbuff_h, recvbuff_h, sendbuff_d, recvbuff_d, count);
  CUDACHECK( cudaStreamCreate(&stream) );
  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday(&tvs, NULL);
  NCCLCHECK( ncclAllGather((void *)(sendbuff_d + sendcount * proc_info.rank),
                           (void *)recvbuff_d, sendcount,
                           NCCL_DTYPE, nccl_info.nccl_comm, stream) );
  gettimeofday(&tve, NULL);
  allgather_check_result(sendbuff_h, recvbuff_h, recvbuff_d, count, proc_info);
  print_result(tvs, tve, count);
}


void allgather_init_values(DTYPE *sendbuff_h, DTYPE *recvbuff_h,
                           DTYPE *sendbuff_d, DTYPE *recvbuff_d, int count) {
  int i = 0;
  for (i=0; i<count; i++) {
    sendbuff_h[i] = 0;
  }
  CUDACHECK( cudaMemcpy(sendbuff_d, sendbuff_h, sizeof(DTYPE) * count,
                        cudaMemcpyHostToDevice) );
  memset((void *)sendbuff_h, 0, sizeof(DTYPE) * count);
  CUDACHECK( cudaMemcpy(sendbuff_d, sendbuff_h, sizeof(DTYPE) * count,
                        cudaMemcpyHostToDevice) );
}


void allgather_check_result(DTYPE *sendbuff_h, DTYPE *recvbuff_h,
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
    if (!(send - eps <= recv &&
          send + eps >= recv)) {
      fprintf(stderr, "rank: %d, value not correct: %f != %f\n", proc_info.rank,
              send, recv);
      exit(EXIT_FAILURE);
    }
  }
}
