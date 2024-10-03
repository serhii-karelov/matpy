#include <stdio.h>
#include "matrix.h"

int _matrix_matmul_4_register_blocked(CMatrix* result, CMatrix* m1, CMatrix* m2);
int main(int argc, char* argv[]) {
  CMatrix* A = NULL;
  CMatrix* B = NULL;
  CMatrix* C = NULL;
  matrix_alloc(&A, 2500, 2500);
  matrix_alloc(&B, 2500, 2500);
  matrix_alloc(&C, 2500, 2500);
  matrix_fill_rand(A, 1);
  matrix_fill_rand(B, 1);
  matrix_matmul(C, A, B);
  // _matrix_matmul_4_register_blocked(C, A, B);
  printf("Done 5 loops\n");
  return 0;
}
