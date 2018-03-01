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


class MyClassName {
  public:
    MyClassName(int n);
    int data_member;
    int member_function();  // Only prototype
  private:
};


MyClassName::MyClassName(int n) {
  data_member = n;
}

int MyClassName::member_function() {
  return data_member;  // data members are accessable this scope
}


int main(int argc, char** argv) {
  MyClassName my_class_name = MyClassName(10);
  std::cout << my_class_name.member_function() << std::endl;
  return 0;
}
