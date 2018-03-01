/********************************
 * Explicit contructor in C++
 * 
 *
********************************/
#include <iostream>

#include "explicit.h"


A::A(int arg) {
  std::cout << "A constructed" << std::endl;
  value_ = arg;
}

int A::get_value() {
  return value_;
}

B::B(int arg) {
  std::cout << "B constructed" << std::endl;
  value_ = arg;
}

int B::get_value() {
  return value_;
}

int main(int argc, char **argv) {
  A a_implicit = 1;  // OK: converting constructor
  A a_explicit = A(2);
  // B b_implicit = 1;  // ERROR: converting converting is disabled by explicit
  B b_explicit = B(2);

  std::cout << a_implicit.get_value() << std::endl;
  std::cout << a_explicit.get_value() << std::endl;
  std::cout << b_explicit.get_value() << std::endl;
  return 0;
}
