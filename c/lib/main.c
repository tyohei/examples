#include "static.h"  // static_func
#include "shared.h"  // shared_func


int main(int argc, char **argv) {
  static_func();
  shared_func();
  default_shared_func();
  // hidden_shared_func(); -> ERROR
  // internal_shared_func(); -> ERROR
  protected_shared_func();
  return 0;
}
