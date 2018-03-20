#include <stdio.h>
#include <stdlib.h>


void inplace_swap(int *x, int *y) {
  *y = *x ^ *y;
  *x = *x ^ *y;
  *y = *x ^ *y;
}


void reverse_array(int *a, int cnt) {
  int first, last;
  for (first=0, last=cnt - 1; first<last; first++, last--) {
    inplace_swap(&a[first], &a[last]);
  }
}


int main(int argc, char **argv) {
  int a[5] = {0, 1, 2, 3, 4};
  printf("[%d, %d, %d, %d, %d]\n", a[0], a[1], a[2], a[3], a[4]);
  reverse_array(a, 5);
  printf("[%d, %d, %d, %d, %d]\n", a[0], a[1], a[2], a[3], a[4]);
  return 0;
}
