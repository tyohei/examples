/********************************
 * Lambda Expression
 *
 *    Lambda expression is a feature that makes easier to define a function
 *    object in C++.
 *    Lambda expression is declared as ``auto`` type.
 *    There are three avilable syntaxes
 *    
 *    ```
 *    [captures](params) -> ret {body}
 *    [captures](params) {body}
 *    [captures] {body}
 *    ```
 *
********************************/
#include <iostream>


int main(int argc, char **argv) {
  /**
   * This is the most standard lambda expression.
   * Here we declare a lambda expression ``f0``, but actually you don't have to
   * declare this.
   */
  auto f0 = [](int x, int y) -> double { return x * y; };
  std::cout << f0(1, 1) << std::endl;
  std::cout << f0(2, 2) << std::endl;
  std::cout << f0(3, 3) << std::endl;
  std::cout << [](int x, int y) -> double { return x * y; }(1, 1) << std::endl;
  std::cout << [](int x, int y) -> double { return x * y; }(2, 2) << std::endl;
  std::cout << [](int x, int y) -> double { return x * y; }(3, 3) << std::endl;


  /**
   * You can omit the return type if you prefer.
   */
  auto f1 = [](int x, int y) { return x * y; };
  std::cout << f1(1, 1) << std::endl;
  std::cout << f1(2, 2) << std::endl;
  std::cout << f1(3, 3) << std::endl;
  std::cout << [](int x, int y) { return x * y; }(1, 1) << std::endl;
  std::cout << [](int x, int y) { return x * y; }(2, 2) << std::endl;
  std::cout << [](int x, int y) { return x * y; }(3, 3) << std::endl;


  /**
   * If there are no arguments, you can omit the arguments also.
   */
  auto f2 = [] { return 0; };
  std::cout << f2() << std::endl;
  std::cout << f2() << std::endl;
  std::cout << f2() << std::endl;


  /* ---------------------------------------------------------------- */
  /**
   * Captures
   *
   *    To access variables outside the lambda expression from the inside, you
   *    need to use **captures**. Captures is declare in the ``[]`` part of the
   *    lambda expression.
   */
  int a3 = 4;
  int b3 = 2;
  // auto f3 = [](int n) { return n * (a3 + b3); };  // -> ERROR

  /**
   * To read outside variables, use captures.
   */
  int a4 = 4;
  int b4 = 2;
  auto f4 = [a4, b4](int n) { return n * (a4 + b4); };
  std::cout << f4(10) << std::endl;

  /**
   * To read and write outside variables, use captures with ``&``.
   */
  int a5 = 4;
  int b5 = 2;
  auto f5 = [&a5, &b5](int n) { a5 = 1; b5 = 2; return n * (a5 + b5); };
  std::cout << f5(10) << std::endl;
  std::cout << "a5: " << a5 << std::endl;
  std::cout << "b5: " << b5 << std::endl;

  /**
   * To access **ALL** variables, use ``=`` in captures.
   */
  int a6 = 4;
  int b6 = 2;
  auto f6 = [=](int n) { return n * (a6 + b6); };
  std::cout << f6(10) << std::endl;

  return 0;
}
