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
