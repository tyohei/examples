/****************************************************************
 * Class in C++
 *
 *    - Class is a extension on struct in C, and it can be said that a class is
 *      a struct that has functions and access atribution.
 *    - Class is composed by data members and member functions.
 *    - Member functions are normally defined **OUTSIDE** of the class
 *      declaration, and the only prototype is written inside the class.
 *    - Usally, the class declaration is written in the header file .h.
 *
****************************************************************/
#include <iostream>

#include "constructor.h"


A::A(int a) {
  a_ = a;
  std::cout << "Construct A" << std::endl;
}

int A::get_a() {
  return a_;
}

B::B(int b) : A(b) {
  b_ = b;
  std::cout << "Construct B" << std::endl;
}

int B::get_b() {
  return b_;
}


int main(int argc, char **argv) {
  A a = A(0);
  B b = B(1);

  std::cout << a.get_a() << std::endl;
  std::cout << b.get_a() << std::endl;
  std::cout << b.get_b() << std::endl;

  return 0;
}
