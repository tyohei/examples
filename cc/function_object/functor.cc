/********************************
 * Functor
 *
 *    Functor is a class which defines the ``operator()`` member function. It's
 *    just a class but you can use it like a function. This is similar to
 *    Python's ``__call__`` method.
 *
********************************/
#include <iostream>


int main(int argc, char **argv) {
  struct Functor0 {
    int operator()(int n) {
      return n * n;
    }
  };
  Functor0 f0 = Functor0();
  std::cout << f0(9) << std::endl;

  return 0;
}
