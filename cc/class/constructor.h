class A {
  public:
    A(int a);
    int get_a();
  private:
    int a_;
};

class B : public A {
  public:
    B(int b);
    int get_b();
  private:
    int b_;
};
