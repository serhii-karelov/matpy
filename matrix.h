#if !defined MATRIX_H
#define MATRIX_H
typedef struct {
  int rows;
  int cols;
  double *items;
} Matrix;

int matrix_alloc(Matrix** m, int rows, int cols);
void matrix_dealloc(Matrix* m);
void matrix_fill(Matrix* m, double x);
void matrix_fill_rand(Matrix* m, int seed);
void matrix_set(Matrix* m, int i, int j, double x);
double matrix_get(Matrix* m, int i, int j);
int matrix_add(Matrix* result, Matrix* m1, Matrix* m2);
int matrix_sub(Matrix* result, Matrix* m1, Matrix* m2);
int matrix_mul(Matrix* result, Matrix* m1, Matrix* m2);
int matrix_matmul(Matrix* result, Matrix* m1, Matrix* m2);
#endif

