/**
 * Reference: TC++PL4 Chapter 16
 */

class X
{
private:
  int m_;
  int n_;
  /**
   * A variable that is part of class, yet it is NOT part of an object of
   * that class, is called a static member.
   * There is exactly one copy of a static member instead of one copy per
   * object.
   */
  static X default_x_;
public:
  explicit X(int m=0, int n=0): m_{m}, n_{n} {}
  int get_m() const {return m_;}
  /**
   * Similarly, a function that needs access to members of a class, yet doesn't
   * need to be invoked for a particular object, is called ``static`` member
   * function.
   */
  static void set_default(int, int);
};


/**
 * A ``static`` member -- a function or data member -- must be defined
 * somewhere (NOT inside the class declaration).
 * The keyword ``static`` is not repeated in the definition of a ``static``
 * member.
 * You must need to add prefix``X::``.
 * In multi-threaded code, ``static`` data members require some kind of
 * locking or access discipline to avoid race conditions.
 */
X X::default_x_ {0, 0};
void X::set_default(int m, int n)
{
  default_x_ = X{m, n};
}


int main() {
  /**
   * A static member can be refered to w/o mentioning an object.
   */
  X::set_default(4, 6);
  X var0 {};
  return 0;
}
