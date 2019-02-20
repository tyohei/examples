long fact_do(long n)
{
  long result = 1;
  do
  {
    result *= n;
    n = n - 1;
  }
  while (n > 1);
  return result;
}


int main(int argc, char **argv)
{
  return 0;
}
