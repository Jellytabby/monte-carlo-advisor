
void loop_add_mul(int idx_a, int idx_b, int idx_c,
                  int a[], int b[], int c[])
{
  c[idx_c] += a[idx_a] * b[idx_b];
}

void loop_add_sub(int idx_a, int idx_b, int idx_c,
                  int a[], int b[], int c[])
{
  c[idx_c] += a[idx_a] - b[idx_b];

}
void loop_sub_sub(int idx_a, int idx_b, int idx_c,
                  int a[], int b[], int c[])
{
  c[idx_c] -= a[idx_a] - b[idx_b];
}

void run_triplet_loop(int length,
                      int a[], int b[], int c[])
{
  for (int i = 0; i < length; i++) {
    for (int j = 0; j < length; j++) {
      for (int k = 0; k < length; k++) {
        loop_add_mul(i, j, k, a, b, c);
        loop_add_sub(i, j, k, a, b, c);
        loop_sub_sub(i, j, k, a, b, c);
        loop_add_mul(i, j, k, a, b, c);
        loop_add_sub(i, j, k, a, b, c);
        loop_sub_sub(i, j, k, a, b, c);
        loop_add_mul(i, j, k, a, b, c);
        loop_add_sub(i, j, k, a, b, c);
        loop_sub_sub(i, j, k, a, b, c);
      }
    }
  }
}

