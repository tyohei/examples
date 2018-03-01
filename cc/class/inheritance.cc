/****************************************************************
 * Inheritance in C++
 *
 *    - Class inheritance in C++ has three types
 *        1. public
 *        2. protected
 *        3. private
 *    - Public inheritance is a relation of **B is a A**.
 *    - Protected inheritance is a relation of **B has a A**.
 *    - Private inheritance is a relation of **B is implemented in terms of
 *      A**.
 *    - The default inheritance type is **private**.
 *
****************************************************************/
#include <iostream>

#include "inheritance.h"


void A::f0() {
  std::cout << "f0()" << std::endl;
}

void A::f1() {
  std::cout << "f0()" << std::endl;
}

void A::f2() {
  std::cout << "f2()" << std::endl;
}

void A0::g0() {
  std::cout << "g0()" << std::endl;
  f0();
  f1();
  // f2();  // ERROR: private function f2 is not accessable
}

void A0::g1() {
  std::cout << "g1()" << std::endl;
  f0();
  f1();
  // f2();  // ERROR: private function f2 is not accessable
}

void A0::g2() {
  std::cout << "g2()" << std::endl;
  f0();
  f1();
  // f2();  // ERROR: private function f2 is not accessable
}

void A1::g0() {
  std::cout << "g0()" << std::endl;
  f0();
  f1();
  // f2();  // ERROR: private function f2 is not accessable
}

void A1::g1() {
  std::cout << "g1()" << std::endl;
  f0();
  f1();
  // f2();  // ERROR: private function f2 is not accessable
}

void A1::g2() {
  std::cout << "g2()" << std::endl;
  f0();
  f1();
  // f2();  // ERROR: private function f2 is not accessable
}

void A2::g0() {
  std::cout << "g0()" << std::endl;
  f0();
  f1();
  // f2();  // ERROR: private function f2 is not accessable
}

void A2::g1() {
  std::cout << "g1()" << std::endl;
  f0();
  f1();
  // f2();  // ERROR: private function f2 is not accessable
}

void A2::g2() {
  std::cout << "g2()" << std::endl;
  f0();
  f1();
  // f2();  // ERROR: private function f2 is not accessable
}

void A00::h0() {
  std::cout << "h0()" << std::endl;
  f0();
  f1();
  // f2();  // ERROR: private function f2 is not accessable
  g0();
  g1();
  // g2();  // ERROR: private function f2 is not accessable
}

void A00::h1() {
  std::cout << "h1()" << std::endl;
  f0();
  f1();
  // f2();  // ERROR: private function f2 is not accessable
  g0();
  g1();
  // g2();  // ERROR: private function f2 is not accessable
}

void A00::h2() {
  std::cout << "h2()" << std::endl;
  f0();
  f1();
  // f2();  // ERROR: private function f2 is not accessable
  g0();
  g1();
  // g2();  // ERROR: private function f2 is not accessable
}

void A10::h0() {
  std::cout << "h0()" << std::endl;
  f0();
  f1();
  // f2();  // ERROR: private function f2 is not accessable
  g0();
  g1();
  // g2();  // ERROR: private function f2 is not accessable
}

void A10::h1() {
  std::cout << "h1()" << std::endl;
  f0();
  f1();
  // f2();  // ERROR: private function f2 is not accessable
  g0();
  g1();
  // g2();  // ERROR: private function f2 is not accessable
}

void A10::h2() {
  std::cout << "h2()" << std::endl;
  f0();
  f1();
  // f2();  // ERROR: private function f2 is not accessable
  g0();
  g1();
  // g2();  // ERROR: private function f2 is not accessable
}

void A20::h0() {
  std::cout << "h0()" << std::endl;
  // f0();  // ERROR: private inheritance change public f0 to private
  // f1();  // ERROR: private inheritance change protected f1 to private
  // f2();  // ERROR: private function f2 is not accessable
  g0();
  g1();
  // g2();  // ERROR: private function f2 is not accessable
}

void A20::h1() {
  std::cout << "h1()" << std::endl;
  // f0();  // ERROR: private inheritance change public f0 to private
  // f1();  // ERROR: private inheritance change protected f1 to private
  // f2();  // ERROR: private function f2 is not accessable
  g0();
  g1();
  // g2();  // ERROR: private function f2 is not accessable
}

void A20::h2() {
  std::cout << "h2()" << std::endl;
  // f0();  // ERROR: private inheritance change public f0 to private
  // f1();  // ERROR: private inheritance change protected f0 to private
  // f2();  // ERROR: private function f2 is not accessable
  g0();
  g1();
  // g2();  // ERROR: private function f2 is not accessable
}


int main(int argc, char **argv) {
  A a_ = A0();  // OK: only public inheritance is allowed
  A b_ = A00();  // OK: only public inheritance is allowed
  A *c_ = new A0();  // OK: only public inheritance is allowed
  A *d_ = new A00();  // OK: only public inheritance is allowed

  // A a = A1();  // ERROR
  // A a = A2();  // ERROR

  A0 a0 = A0();
  A1 a1 = A1();
  A2 a2 = A2();

  A00 a00 = A00();
  A10 a10 = A10();
  A20 a20 = A20();


  a0.f0();  // OK
  // a1.f0();  // ERROR: protected inheritance change public f0 to protected
  // a2.f0();  // ERROR: private inheritance change public f0 to private

  // a0.f1();  // ERROR: protected function f1 is not accessable
  // a1.f1();  // ERROR: same as above
  // a2.f1();  // ERROR: same as above

  // a0.f2();  // ERROR: private function f2 is not accessable
  // a1.f2();  // ERROR: same as above
  // a2.f2();  // ERROR: same as above


  a0.g0();  // OK
  a1.g0();  // OK
  a2.g0();  // OK

  // a0.g1();  // ERROR: protected function g1 is not accessable
  // a1.g1();  // ERROR: same as above
  // a2.g1();  // ERROR: same as above

  // a0.g2();  // ERROR: private function g2 is not accessable
  // a1.g2();  // ERROR: same as above
  // a2.g2();  // ERROR: same as above


  a00.g0();  // OK
  a10.g0();  // OK
  a20.g0();  // OK

  // a00.g1();  // ERROR: protected function f1 is not accessable
  // a10.g1();  // ERROR: protected function f1 is not accessable
  // a20.g1();  // ERROR: protected function f1 is not accessable

  // a00.g2();  // ERROR: private function f2 is not accessable;
  // a10.g2();  // ERROR: private function f2 is not accessable;
  // a20.g2();  // ERROR: private function f2 is not accessable;
  

  a00.h0();  // OK
  a10.h0();  // OK
  a20.h0();  // OK

  // a00.h1();  // ERROR: protected function f1 is not accessable
  // a10.h1();  // ERROR: protected function f1 is not accessable
  // a20.h1();  // ERROR: protected function f1 is not accessable

  // a00.h2();  // ERROR: private function f1 is not accessable
  // a10.h2();  // ERROR: private function f1 is not accessable
  // a20.h2();  // ERROR: private function f1 is not accessable

  return 0;
}
