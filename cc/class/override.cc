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

#include "override.h"


A::A() {
  std::cout << "Construct A" << std::endl;
}

void A::f() {
  std::cout << "f() in A" << std::endl;
}

void A::g() {
  std::cout << "g() in A" << std::endl;
}

void A::h() {
  std::cout << "h() in A" << std::endl;
}

B::B() {
  std::cout << "Construct B" << std::endl;
}

void B::f() {
  std::cout << "f() in B" << std::endl;
}

void B::g() {
  std::cout << "g() in B" << std::endl;
}

void B::h() {
  std::cout << "h() in B" << std::endl;
}

void B::f_B() {
  std::cout << "f_B() in B" << std::endl;
}


int main(int argc, char **argv) {
  std::cout << "A a = A()" << std::endl; A a = A();
  std::cout << "A b = B()" << std::endl; A b = B();
  std::cout << std::endl;
  std::cout << "A *a_ = new A()" << std::endl; A *a_ = new A();
  std::cout << "A *b_ = new B()" << std::endl; A *b_ = new B();
  std::cout << std::endl;
  std::cout << "B actual_B = B()" << std::endl; B actual_B = B();

  std::cout << "a.f(): "; a.f();
  std::cout << "a.g(): "; a.g();
  std::cout << "a.h(): "; a.h();

  std::cout << "b.g(): "; b.f();
  std::cout << "b.g(): "; b.g();
  std::cout << "b.g(): "; b.h();
  // b.f_B();  // ERROR: b is instance of A 


  std::cout << "a_->f(): "; a_->f();
  std::cout << "a_->g(): "; a_->g();
  std::cout << "a_->h(): "; a_->h();

  std::cout << "b_->f(): "; b_->f();
  std::cout << "b_->g(): "; b_->g();
  std::cout << "b_->h(): "; b_->h();
  // b_->f_B();  // ERROR: b_ is instance of A 

  std::cout << "actual_B.f(): "; actual_B.f();
  std::cout << "actual_B.g(): "; actual_B.g();
  std::cout << "actual_B.h(): "; actual_B.h();
  std::cout << "actual_B.f_B(): "; actual_B.f_B();
  return 0;
}
