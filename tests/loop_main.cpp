#include <iostream>

void loop_add_mul(int, int, int, int[], int[], int[]);
void run_triplet_loop(int, int[], int[], int[]);
extern "C" void __mc_profiling_begin(void);
extern "C" void __mc_profiling_end(void);

int main() {
  int length = 200;
  int *a = new int[length];
  int *b = new int[length];
  int *c = new int[length](); // zero‐initialize c[]

  std::cout << "Your current length is: " << length << '\n';

  for (int i = 0; i < length; i++) {
    a[i] = i;
    b[i] = length - i;
  }

  __mc_profiling_begin();
  run_triplet_loop(length, a, b, c);
  __mc_profiling_end();

  std::cout << "FINISHED THE LOOP\n";

  delete[] a;
  delete[] b;
  delete[] c;
  return 0;
}
