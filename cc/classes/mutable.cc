/**
 * Reference: TC++PL4 Chapter 16
 */

struct cache {
  bool i;
};

class X
{
private:
  /**
   * We can define a member of a class to be mutable, meaning that it can be
   * modified event in const object (here get_m())
   * Declaring a member ``mutable`` is most appropriate whenonly a small part
   * of a representation of a small object is allowed to cahnge.
   * More complicated cases are often better handled by placing the changing
   * data in a separate object and accessing it indirectly.
   */
  mutable int m_;
  int n_;
  // const does not apply to objects accessed through pointers or references.
  cache *c_;
public:
  explicit X(int m=0, int n=0): m_{m}, n_{n} {}
  int get_m(int) const;
};


int X::get_m(int i) const
{
  m_++;  // OK, m_ is mutable
  // n_++;  // ERROR, n_ is NOT mutable
  c_->i = true;  // OK c_ is a pointer
  return m_ + i;
}


int main() {
  X var0 {};
  return 0;
}
