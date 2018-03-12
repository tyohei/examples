#include <iostream>


template <typename T>
T twice(T a) { return 2 * a; }
// This is same as 
/*
 *    >>> template <class T>
 *    >>> T twice(T a) { return 2 * a; }
 *
 */
// Explicit instantation
//    If you are dividing the declaration and implementation into different
//    files (e.g *.h file and *.cc file), you MUST write an instantation
//    explicitly in the implementation file. Otherwise you will get a linker
//    error.
// ================================
// template twice<int>;   // -> ERROR
// int twice<int>(int);   // -> ERROR
// int twice(int);        // -> ERROR (linker)
template int   twice       (int);    // OK
template float twice<float>(float);  // OK



int main(int argc, char **argv) {
  std::cout << twice(1) << std::endl;
  return 0;
}
