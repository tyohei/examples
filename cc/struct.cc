/********************************
 * Structure in C++
 * 
 *    In C++ structure is exactly same as class except the visibility is
 *    ``public`` in ``struct`` and ``private`` in ``class``. That means you can
 *    use member functions and inheritance in structure.
 *    With regard to the difference between C structure, C++ structure does not
 *    need to use ``typedef``.
 *
********************************/
#include <iostream>


int main(int argc, char **argv) {

  /**
   * This is NOT permitted in C++, dispite it is permitted in C.
   */
  // struct {
  //   int a;
  //   int b;
  // };
  // -> ERROR

  struct {
    int a;
    int b;
  } structure_0;


  struct Hoge {
    int a;
    int b;
  };
  Hoge hoge = Hoge();
  struct Hoge hoge1;
  hoge.a = 1;
  hoge.b = 2;
  std::cout << hoge.a << std::endl;
  std::cout << hoge.b << std::endl;
  hoge1.a = 2;
  hoge1.b = 4;
  std::cout << hoge1.a << std::endl;
  std::cout << hoge1.b << std::endl;


  return 0;
} 
