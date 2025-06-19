
void loop_add_mul(int length, int idx_a, int idx_b, int a[], int b[], int c[]) {
  for (int k = 0; k < length; k++) {
    c[k] = a[idx_a] * b[idx_b];
  }
}

void run_triplet_loop(int length, int a[], int b[], int c[]) {
  for (int i = 0; i < length; i++) {
    for (int j = 0; j < length; j++) {
      loop_add_mul(length, i, j, a, b, c);
    }
  }
}
