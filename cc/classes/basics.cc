/**
 * Reference: TC++PL4 Chapter 16
 * ================================
 *  - A class is a user-defined type.
 *  - A class consists of a set of members. The most common kinds of members
 *    are data members and member functions.
 *  - Member functions can define the meaning of initialization (creation),
 *    copy, move, and cleanup (destruction).
 *  - Members are accessed using . (dot) for objects and -> (arrow) for pointers.
 *  - Operators, such as +, !, and [], can be defined for a class.
 *  - A class is a namespace containing its members.
 *  - The public members provide the class's interface and the private members
 *    provide implementation details.
 *  - A struct is a class where member are by default public.
 * ================================
 */

/**
 * The construct is called a class definition
 */
class X
{
  // By default, members of a class is private
private:
  int m_;
public:
  // A constructor is recognized by having the same name as the class iteslf.
  // You can provide several constructors.
  X(): m_{0} {}
  X(int i): m_{i} {}
  X(int i, int j): m_{i} {}

  // Functions declared within a class definition are called member functions
  int mf(int i)
  {
    int old = m_;
    m_ = i;
    return old;
  }
};


int user(X var, X *ptr)
{
  int x = var.mf(7);
  int y = ptr->mf(9);
}


int main() {
  X var0(9);
  X var1 {7};
  /**
   * Author of TC++PL3 recommends to use the {} notation over the () notation
   * for initialization. Because it is explicit about what is being done,
   * INITIALIZATION, avoids some potential mistakes.
   */
  X var2 {var1};  // Objects can be copied
  X var3 = var1;  // Objects can be copied
  /* By default, the copy of a class object is a copy of EACH member, you can
   * provied a different behavior if you want. */
  return 0;
}
