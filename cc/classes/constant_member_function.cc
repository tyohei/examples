/**
 * Reference: TC++PL4 Chapter 16
 */

class X
{
private:
  int m_;
  int n_;
public:
  explicit X(int m=0, int n=0): m_{m}, n_{n} {}
  /**
   * The ``const`` after the  argument list in the function declarations
   * indicates that these functions do not modify the state of X.
   */
  int get_m(int) const;
};


int X::get_m(int i) const
{
  // m_++;  // ERROR, attempt to change member value
  return m_ + i;
}


int main() {
  X var0 {};
  return 0;
}
