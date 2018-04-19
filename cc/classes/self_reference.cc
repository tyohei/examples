/**
 * Reference: TC++PL4 Chapter 16
 */
#include <iostream>


class X
{
private:
  int m_;
  int n_;
public:
  explicit X(int m=0, int n=0): m_{m}, n_{n} {}
  X &inc_m(int);
  int get_m() const { return m_; }
};


X &X::inc_m(int i)
{
  /**
   * Each (non-static) member function knows for which object is was invoked
   * and can explicitly refer to it.
   */
  m_ += i;  // same as below
  // this->m_ += i;
  /**
   * The expr ``*this`` refers to the object for which a member is invoked.
   * In a non-static member function, the keyword ``this`` is a POINTER to the
   * object for which the function was invoked.
   * In a non-const member function of class X, the type of ``this`` is ``*X``,
   * however, ``this`` is considered an ___rvalue___.
   * In a const member function of class X, the type of ``this`` is
   * ``const *X``.
   */
  return *this;
}


int main() {
  X var0 {};
  var0.inc_m(1).inc_m(2);
  std::cout << var0.get_m() << std::endl;
  return 0;
}
