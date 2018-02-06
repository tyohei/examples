/********************************
 * Function Pointer
 *
 *    Function pointer is a pointer that points a function. It is useful when
 *    you considering to use C/C++ together. The syntax is like below (only
 *    show a function pointer that takes two arguments).
 *    
 *    ```
 *    returntype (*name)(argtype, argtype);
 *    ```
 *
********************************/
#include <iostream>


int f(int a, int b) {
  return a * b;
}


int wrapper(int n, int(*func)(int, int)) {
  return func(n, n);
}


int main(int argc, char **argv) {

  int (*f_ptr)(int, int);

  f_ptr = &f;

  std::cout << (*f_ptr)(9, 9) << std::endl;
  std::cout << f_ptr(4, 4) << std::endl;

  std::cout << wrapper(9, f) << std::endl;
  std::cout << wrapper(10, f) << std::endl;

  return 0;
}
