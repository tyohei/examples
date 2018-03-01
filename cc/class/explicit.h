class A {
  public:
    A(int arg);
    int get_value();
  private:
    int value_;
};


class B {
  public:
    explicit B(int arg);
    int get_value();
  private:
    int value_;
};
