#include <stdio.h>    // printf
#include <stdlib.h>   // malloc, calloc
#include <time.h>     // time, clock


void use_malloc(size_t n)
{
  time_t bt = time(NULL);
  float *arr = (float *)malloc(n * sizeof(float));
  float sum = 0.0;
  for (int i=0; i<n; ++i)
    sum += arr[i];
  time_t ft = time(NULL);
  printf("sum: %f, malloc time: %lu\n", sum, ft - bt);
  free(arr);
  return;
}


void use_calloc(size_t n)
{
  time_t bt = time(NULL);
  float *arr = (float *)calloc(n, sizeof(float));
  float sum = 0.0;
  for (int i=0; i<n; ++i)
    sum += arr[i];
  time_t ft = time(NULL);
  printf("sum: %f, calloc time: %lu\n", sum,  ft - bt);
  free(arr);
  return;
}


void use_alloca(size_t n)
{
  time_t bt = time(NULL);
  float *arr = (float *)alloca(n * sizeof(float));
  float sum = 0.0;
  for (int i=0; i<n; ++i)
    sum += arr[i];
  time_t ft = time(NULL);
  printf("sum: %f, alloca time: %lu\n", sum,  ft - bt);
  return;
}


int main()
{
  size_t n = 1024 * 1024 * 1024;  // 1 GiB
  int ntest = 2;
  for (int i=0; i<ntest; ++i)
    use_malloc(n);
  for (int i=0; i<ntest; ++i)
    use_calloc(n);
  /*  This causes segfault due to stack overflow!!!
   *  which means ``alloca`` is BAD to use in codes
  for (int i=0; i<ntest; ++i)
    use_alloca(n);
  */
  return 0;
}
