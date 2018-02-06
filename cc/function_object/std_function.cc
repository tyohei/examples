/********************************
 * ``std::function``
 *
 *    Functor is a class which defines the ``operator()`` member function. It's
 *    just a class but you can use it like a function. This is similar to
 *    Python's ``__call__`` method.
 *
********************************/
#include <iostream>
#include <functional>


int func0(int n) {
  return n * n;
}


struct Functor {
  int operator()(int n) {
    return n * n;
  }
};


int main(int argc, char **argv) {
  std::function< int(int) > f0 = &func0;
  std::cout << f0(10) << std::endl;

  std::function< int(int) > f1 = Functor();
  std::cout << f1(9) << std::endl;

  return 0;
}
