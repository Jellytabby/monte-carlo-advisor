void run_recursions(int);

extern "C" void __mc_inline_begin(void);
extern "C" void __mc_inline_end(void);

int main() {
  int n = 2000000;

  __mc_inline_begin();
  for (int i = 0; i < n; i++) {
    run_recursions(i);
  }
  __mc_inline_end();

  return 0;
}
