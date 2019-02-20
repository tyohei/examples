#include <stdio.h>


/* Example of for loop containing a continue statement */
/* Sum even numbers between 0 and 9 */
int main()
{
  long sum = 0;
  long i;
  for (i = 0; i < 10; i++)
  {
    if (i & 1)
    {
      continue;
    }
    sum += i;
  }
  printf("%ld\n", sum);

  /* A */
  /* Naive translation to while loop */
  /* THIS CODE WILL CAUSE INFINITY-LOOP!!! */
#if 0
  sum = 0;
  /* init-expr */
  i = 0;
  /* test-expr */
  while (i < 10)
  {
    /* body-statement */
    if (i & 1)
    {
      continue;
    }
    sum += i;
    /* update-expr */
    i++;
  }
  printf("%ld\n", sum);
  /**
   * The problem of this code is update-expr is not called when the
   * continue statement is called.
   */
#endif  // 0

  /* B */
  sum = 0;
  for (i = 0; i < 10; i++)
  {
    if (i & 1)
    {
      goto next1;
    }
    sum += i;
next1:
    ;
  }
  printf("%ld\n", sum);

  /* while version */
  sum = 0;
  /* init-expr */
  i = 0;
  /* test-expr */
  while (i < 10)
  {
    if (i & 1)
    {
      goto next2;
    }
    sum += i;
next2:
    ;
    /* update-expr */
    i++;
  }
  printf("%ld\n", sum);
}
