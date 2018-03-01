#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>


int main(int argc, char **argv) {
  double *buf_d = NULL;
  fprintf(stderr, "Allocating...\n");
  cudaMalloc((void **) &buf_d, sizeof(double) * 1024);
  fprintf(stderr, "Allocating DONE.\n");
  return 0;
}
