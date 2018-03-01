class A {
  public:
    void f0();
  protected:
    void f1();
  private:
    void f2();
};


class A0 : public A {
  public:
    void g0();
  protected:
    void g1();
  private:
    void g2();
};


class A1 : protected A {
  public:
    void g0();
  protected:
    void g1();
  private:
    void g2();
};


class A2 : private A {
  public:
    void g0();
  protected:
    void g1();
  private:
    void g2();
};


class A00 : public A0 {
  public:
    void h0();
  protected:
    void h1();
  private:
    void h2();
};


class A10 : public A1 {
  public:
    void h0();
  protected:
    void h1();
  private:
    void h2();
};


class A20 : public A2 {
  public:
    void h0();
  protected:
    void h1();
  private:
    void h2();
};
