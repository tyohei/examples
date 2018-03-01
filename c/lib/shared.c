#include <stdio.h>

#include "shared.h"

void shared_func() {
  printf("This is a demo of shared library.\n");
}

__attribute__((__visibility__("default"))) void default_shared_func() {
  printf("This is a demo of DEFAULT shared library.\n");
}

__attribute__((__visibility__("hidden"))) void hidden_shared_func() {
  printf("This is a demo of HIDDEN shared library.\n");
}

__attribute__((__visibility__("internal"))) void internal_shared_func() {
  printf("This is a demo of INTERNAL shared library.\n");
}

__attribute__((__visibility__("protected"))) void protected_shared_func() {
  printf("This is a demo of PROTECTED shared library.\n");
}
