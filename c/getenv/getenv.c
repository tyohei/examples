#include <stdio.h>
#include <stdlib.h>  // getenv()


int main(int argc, char **argv) {
  char *shell = NULL;

  shell = getenv("SHELL");

  printf("SHELL: %s\n", shell);

  return 0;
}
