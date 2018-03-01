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

#include "virtual.h"


void A::f() {
  std::cout << "f() in A" << std::endl;
}

void A::g() {
  std::cout << "g() in A" << std::endl;
}

void B::f() {
  std::cout << "f() in B" << std::endl;
}

void B::g() {
  std::cout << "g() in B" << std::endl;
}


int main(int argc, char **argv) {
  A *a = new A();
  A *b = new B();
  A a_ = A();
  A b_ = B();

  a->f();  // f() in A
  a->g();  // g() in A
  b->f();  // f() in A
  b->g();  // g() in B

  a_.f();  // f() in A
  a_.g();  // g() in A
  b_.f();  // f() in A
  b_.g();  // g() in A
  return 0;
}
