extern "C" void __mc_inline_begin(void);
extern "C" void __mc_inline_end(void);

void run_recursions(int);

int main() {
  int n = 30;

  __mc_inline_begin();
  run_recursions(n);
  __mc_inline_end();

  return 0;
}
