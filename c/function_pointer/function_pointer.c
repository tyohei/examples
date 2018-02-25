#include <stdio.h>
#include <stdint.h>

typedef int int_0;
typedef int int_1;
typedef int int_2;
typedef int *(s_t)(int, int);

int add(int a, int b) {
  return a + b;
}

int sub(int a, int b) {
  return a - b;
}

int (*arith(int a, int b))(int, int) {
  switch (a) {
    case 0:
      return &add;
    default:
      return &sub;
  }
}

int main(int argc, char **argv) {

  // Function pointer that takes two arguments
  //    int_0
  //    int_0
  // And returns
  //    int
  int (*f_ptr)(int_0, int_0);

  // Function pointer that takes two arguments
  //    int_0
  //    int_0
  // And returns
  //    Funtion pointer that takes two arguments
  //        int_1
  //        int_1
  //    And returns
  //        int
  int ( *(*g_ptr)(int_0, int_0) )(int_1, int_1);

  // Function pointer that takes two arguments
  //    int_0
  //    int_0
  // And returns
  //    Funtion pointer that takes two arguments
  //        int_1
  //        int_1
  //    And returns
  //        Funtion pointer that takes two arguments
  //            int_2
  //            int_2
  //        And returns
  //            int
  int ( *(*(*h_ptr)(int_0, int_0))(int_1, int_1) )(int_2, int_2);

  // Funtion pointer that takes two arguments
  //    Funtion pointer that takes one arguments
  //        int
  //    Funtion pointer that takes one arguments
  //        int
  // And returns
  //        int
  int (*r_ptr)(int (*)(int), int (*)(int));


  printf("add:      %p\n", add);
  printf("sub:      %p\n", sub);
  printf("main:     %p\n", main);
  printf("sp:       %p\n", __builtin_frame_address(0));

  printf("NULL    : %p\n", NULL);

  printf("f_ptr:    %p\n", f_ptr);
  printf("g_ptr:    %p\n", f_ptr);
  printf("h_ptr:    %p\n", f_ptr);

  f_ptr = &add;
  printf("f_ptr = &add;\n");
  printf("f_ptr(23, 13) = %d\n", f_ptr(23, 13));
  printf("\n");
  f_ptr = &sub;
  printf("f_ptr = &sub;\n");
  printf("f_ptr(23, 13) = %d\n", f_ptr(23, 13));
  printf("\n");

  g_ptr = &arith;
  printf("g_ptr(0, 2)(23, 13) = %d\n", g_ptr(0, 2)(23, 13));
  printf("g_ptr(1, 2)(23, 13) = %d\n", g_ptr(1, 2)(23, 13));

  return 0;
}
