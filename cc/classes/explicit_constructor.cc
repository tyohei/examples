/**
 * Reference: TC++PL4 Chapter 16
 */
class X
{
private:
  int m_;
  int n_;
public:
  X(int m=0, int n=0): m_{m}, n_{n} {}
};


/**
 * Better than X
 */
class Y
{
private:
  int m_;
  int n_;
public:
  explicit Y(int m=0, int n=0): m_{m}, n_{n} {}
};


int main()
{
  /**
   * By default, a constructor invoked by a SINGLE argument acts as an implicit
   * conversion from its argument type to its type.
   * Such implicit conversions can be extremely useful.
   * However, in many cases, such conversions can be a significant source of
   * confusion and errors.
   */
  X xvar0 {10};
  X xvar1 = 10;  // Implicit conversion from int to X.
  /**
   * Fortunatelly, we can specify that a constructor is not used as an
   * implicit conversion.
   * A constructor declared with the keyword ___EXPLICIT___ can only be used
   * for initialization and explicit conversions.
   * It is recommented to declare explicit before constructor, and if not,
   * you need a good reason not to do so.
   */
  Y yvar0 {10};  // OK
  // Y yvar1 = 10;  // ERROR, = does not do implicit conversions
}
