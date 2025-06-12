int fibonacci(int n) {
  return n <= 1 ? 1 : fibonacci(n - 1) + fibonacci(n - 1);
}

int factorial(int n) { return n <= 1 ? 1 : n * factorial(n - 1); }

void run_recursions(int n) {
  fibonacci(n);
  factorial(n);
}
