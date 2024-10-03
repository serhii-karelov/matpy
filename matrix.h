#if !defined MATRIX_H
#define MATRIX_H
typedef struct {
  int rows;
  int cols;
  double *items;
} CMatrix;

int matrix_alloc(CMatrix** m, int rows, int cols);
void matrix_dealloc(CMatrix* m);
void matrix_fill(CMatrix* m, double x);
void matrix_fill_rand(CMatrix* m, int seed);
void matrix_set(CMatrix* m, int i, int j, double x);
double matrix_get(CMatrix* m, int i, int j);
int matrix_add(CMatrix* result, CMatrix* m1, CMatrix* m2);
int matrix_sub(CMatrix* result, CMatrix* m1, CMatrix* m2);
int matrix_mul(CMatrix* result, CMatrix* m1, CMatrix* m2);
int matrix_matmul(CMatrix* result, CMatrix* m1, CMatrix* m2);
#endif

