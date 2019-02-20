#include <stdio.h>


long absdiff(long x, long y)
{
  long result;
  if (x < y)
    result = y - x;
  else
    result = x - y;
  return result;
}


int main(int argc, char **argv)
{
  return 0;
}
