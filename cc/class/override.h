class A {
  public:
    A();
    void f();
    virtual void g();
    virtual void h();
};

class B : public A {
  public:
    B();
    void f();
    // void f() override;  // ERROR: override MUST be on virtual function
    virtual void g();
    virtual void h() override;
    // virtual void i() override {};  // ERROR: MUST override something
    void f_B();
};
