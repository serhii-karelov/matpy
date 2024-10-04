#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <omp.h>
#include <Python.h> /* PyMem_RawMalloc, PyMem_RawCalloc, PyMem_RawFree */
#include "matrix.h" /* typedef CMatrix */
#include <immintrin.h>

#define MATRIX_ITEM(m, i, j) ((m)->items)[(i) + (j) * (m)->rows]
#define ITEM(items, ld, i, j) (items)[(i) + ld * (j)] 
#define MATRIX_ADDSET(m, i, j, x) ((m)->items)[(i) + (j) * (m)->rows] += x
#define min(a, b) a < b ? a : b
/* micro-kernel dimensions */
#define b_m 12 
#define b_n 4

int matrix_alloc(CMatrix **m, int rows, int cols) {
  (*m) = (CMatrix*) PyMem_RawMalloc(sizeof(CMatrix));
  if ((*m) == NULL) {
    return -1;
  }
  (*m)->rows = rows;
  (*m)->cols = cols;
  (*m)->items = (double*) PyMem_RawCalloc(rows * cols, sizeof(double));
  return 0;
}

void matrix_dealloc(CMatrix *m) {
  if (m == NULL) {
    return;
  }
  if (m->items != NULL) {
     PyMem_RawFree(m->items);
  }
  PyMem_RawFree(m);
}

void matrix_fill(CMatrix *m, double x) {
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      MATRIX_ITEM(m, i, j) = x;
    } 
  }
}

void matrix_fill_rand(CMatrix *m, int seed) {
  srand(seed);
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      MATRIX_ITEM(m, i, j) =  ((double) random()) / 2147483647;
    } 
  }
}

void matrix_set(CMatrix *m, int i, int j, double x) {
  MATRIX_ITEM(m, i, j) = x;
}

double matrix_get(CMatrix *m, int i, int j) {
  return MATRIX_ITEM(m, i, j);
}

static int check_dimensions(CMatrix *m1, CMatrix *m2, CMatrix *m3) {
  return ((m1->rows == m2->rows)
       && (m1->rows == m3->rows)
       && (m1->cols == m2->cols)
       && (m1->cols == m3->cols));
}

static double matrix_elementwise(CMatrix *result, CMatrix *m1, CMatrix *m2, double (*op)(double, double)) {
  if (!check_dimensions(result, m1, m2)) {
    return -1;
  }
  for (int i = 0; i < m1->rows; i++) {
    for (int j = 0; j < m2->cols; j++) {
      MATRIX_ITEM(result, i, j) = (*op)(MATRIX_ITEM(m1, i, j), MATRIX_ITEM(m2, i, j));
    }
  }
  return 0;
}

static double op_add(double a, double b) {
  return a + b;
}

static double op_sub(double a, double b) {
  return a - b;
}

static double op_mul(double a, double b) {
  return a * b;
}

int matrix_add(CMatrix *result, CMatrix *m1, CMatrix *m2) {
  return matrix_elementwise(result, m1, m2, op_add);
}
  
int matrix_sub(CMatrix *result, CMatrix *m1, CMatrix *m2) {
  return matrix_elementwise(result, m1, m2, op_sub);
}

int matrix_mul(CMatrix *result, CMatrix *m1, CMatrix *m2) {
  return matrix_elementwise(result, m1, m2, op_mul);
}

int matrix_matmul(CMatrix *result, CMatrix *m1, CMatrix *m2) {
  return  _matrix_matmul_5_loops(result, m1, m2);
}
int _matrix_matmul_1_ikj(CMatrix *result, CMatrix *m1, CMatrix *m2) {
  if (!((result->rows == m1->rows)
        && (result->cols == m2->cols)
        && (m1->cols == m2->rows))) {
    return -1;
  }
  double a, b;
  int m = m1->rows;
  int n = m2->cols;
  int r = m2->rows;

  double *A = m1->items;
  double *B = m2->items; 
  double *C = result->items;

  for (int i = 0; i < m; i++) {
    for (int k = 0; k < r; k++) {
      a = ITEM(A, m, i, k);
      for (int j = 0; j < n; j++) {
        b = ITEM(B, r, k, j);
        ITEM(C, m, i, j) += a * b;
      }
    }
  }
  return 0;
}

int _matrix_matmul_2_jki(CMatrix *result, CMatrix *m1, CMatrix *m2) {
  if (!((result->rows == m1->rows)
        && (result->cols == m2->cols)
        && (m1->cols == m2->rows))) {
    return -1;
  }
  double a, b;
  int m = m1->rows; // ldA, ldC
  int n = m2->cols; 
  int r = m2->rows; // ldB

  double *A = m1->items;
  double *B = m2->items; 
  double *C = result->items;

  for (int k = 0; k < n; k++) {
    for (int j = 0; j < r; j++) {
      b = ITEM(B, n, k, j);
      for (int i = 0; i < m; i++) {
        a = ITEM(A, m, i, k);
        ITEM(C, m, i, j) += a * b;
      }
    }
  }
  return 0;
}

int _matrix_matmul_2_kji(CMatrix *result, CMatrix *m1, CMatrix *m2) {
  if (!((result->rows == m1->rows)
        && (result->cols == m2->cols)
        && (m1->cols == m2->rows))) {
    return -1;
  }
  double a, b;
  int k;
  for (k = 0; k < m2->cols; k++) {
    for (int j = 0; j < m1->cols; j++) {
      b = MATRIX_ITEM(m2, j, k);
      for (int i = 0; i < m1->rows; i++) {
        a = MATRIX_ITEM(m1, i, j);
        MATRIX_ADDSET(result, i, k, a * b);
      }
    }
  }
  return 0;
}

int _matrix_matmul_3_kji_unrolled(CMatrix *result, CMatrix *m1, CMatrix *m2) {
  if (!((result->rows == m1->rows)
        && (result->cols == m2->cols)
        && (m1->cols == m2->rows))) {
    return -1;
  }
  double a[4], b;
  int i;
  for (int k = 0; k < m2->cols; k++) {
    for (int j = 0; j < m1->cols; j++) {
      b = MATRIX_ITEM(m2, j, k);
      for (i = 0; i < m1->rows - 4; i += 4) {
        a[0] = MATRIX_ITEM(m1, i, j);
        a[1] = MATRIX_ITEM(m1, i + 1, j);
        a[2] = MATRIX_ITEM(m1, i + 2, j);
        a[3] = MATRIX_ITEM(m1, i + 3, j);
        MATRIX_ADDSET(result, i, k,     a[0] * b);
        MATRIX_ADDSET(result, i + 1, k, a[1] * b);
        MATRIX_ADDSET(result, i + 2, k, a[2] * b);
        MATRIX_ADDSET(result, i + 3, k, a[3] * b);
      }
      for (; i < m1->rows; i++) {
        a[0] = MATRIX_ITEM(m1, i, j);
        MATRIX_ADDSET(result, i, k, a[0] * b);
      } 
    }
  }
  return 0;
}


void _matmul_block(int m, int n, int r, int ldA, int ldB, int ldC, double *res, double *block1, double *block2);

int _matrix_matmul_5_loops(CMatrix *result, CMatrix *m1, CMatrix *m2) {
  if (!((result->rows == m1->rows)
        && (result->cols == m2->cols)
        && (m1->cols == m2->rows))) {
    return -1;
  }
  int m = m1->rows / b_m * b_m;
  int n = m2->cols / b_n * b_n;
  int r = m1->cols;
  int ldA = m1->rows;
  int ldB = m2->rows;
  int ldC = result->rows; // same as ldA
  double *A = m1->items;
  double *B = m2->items;
  double *C = result->items;
  _loop_5(
      m, n, r, 
      ldA, A, ldB, B, ldC, C);

  _matmul_remainder(m1->rows / b_m * b_m, m2->cols / b_n * b_n, result, m1, m2);
}

void _loop_5(int m, int n, int r,
             int ldA, double *A, int ldB, double *B, int ldC, double *C) {
  int stride = 128;
#pragma omp parallel for
  for (int j = 0; j < n; j += stride) {
    int jb = min(stride, n - j); // block size for j dimension
    _loop_4(m, jb, r, ldA, A, ldB, &ITEM(B, ldB, 0, j), ldC, &ITEM(C, ldC, 0, j));
  }
}

void _loop_4(int m, int n, int r,
             int ldA, double *A, int ldB, double *B, int ldC, double *C) {
  int stride = 256;
  for (int k = 0; k < r; k += stride) {
    int kb = min(stride, r - k); // block size for k dimension
    _loop_3(m, n, kb, ldA, &ITEM(A, ldA, 0, k), ldB, &ITEM(B, ldB, k, 0), ldC, C);
  }
}

void _loop_3(int m, int n, int r,
             int ldA, double *A, int ldB, double *B, int ldC, double *C) {
  int stride = 48;
  for (int i = 0; i < m; i += stride) {
    int ib = min(stride, m - i); // block size for i dimension
    _loop_2(ib, n, r, ldA, &ITEM(A, ldA, i, 0), ldB, B, ldC, &ITEM(C, ldC, i, 0));
  }
}

void _loop_2(int m, int n, int r,
             int ldA, double *A, int ldB, double *B, int ldC, double *C) {
  int stride = 4;
  for (int j = 0; j < n; j += stride) {
    int jb = min(stride, n - j); // block size for j dimension
    _loop_1(m, jb, r, ldA, A, ldB, &ITEM(B, ldB, 0, j), ldC, &ITEM(C, ldC, 0, j));
  }
}

void _loop_1(int m, int n, int r,
             int ldA, double *A, int ldB, double *B, int ldC, double *C) {
  int stride = 12;
  for (int i = 0; i < m; i += stride) {
     _matmul_kernel_12x4(r, &ITEM(A, ldA, i, 0), ldA, B, ldB, &ITEM(C, ldC, i, 0), ldC);
  }
}

void _matmul_remainder(int start_i, int start_j, CMatrix *result, CMatrix *m1, CMatrix *m2) {
  if (start_i < m1->rows) {
    for (int j = 0; j < result->cols; j++) {
      for (int k = 0; k < m2->rows; k++) {
        for (int i = start_i; i < m1->rows; i++) {
          MATRIX_ITEM(result, i, j) += (MATRIX_ITEM(m1, i, k)) * MATRIX_ITEM(m2, k, j);
        }
      }
    }
  }
  if (start_j < m2->cols) {
    for (int j = start_j; j < result->cols; j++) {
      for (int k = 0; k < m2->rows; k++) {
        for (int i = 0; i < m1->rows / b_m * b_m; i++) {
          MATRIX_ITEM(result, i, j) += (MATRIX_ITEM(m1, i, k)) * MATRIX_ITEM(m2, k, j);
        }
      }
    }
  }
}

int _matrix_matmul_4_register_blocked(CMatrix *result, CMatrix *m1, CMatrix *m2) {
  // Use alternative notation: C_ij += A_ik * B_kj
  if (!((result->rows == m1->rows)
        && (result->cols == m2->cols)
        && (m1->cols == m2->rows))) {
    return -1;
  }
  double *a;
  double *b;
  for (int i = 0; i < m1->rows / b_m * b_m; i += b_m) {
#pragma omp parallel for
    for (int j = 0; j < m2->cols / b_n * b_n; j += b_n)  {
      a = &MATRIX_ITEM(m1, i, 0);
      b = &MATRIX_ITEM(m2, 0, j);
      _matmul_kernel_12x4(m1->cols, a, m1->rows, b, m2->rows,  &MATRIX_ITEM(result, i, j), result->rows); 
    }
  }
  _matmul_remainder(m1->rows / b_m * b_m, m2->cols / b_n * b_n, result, m1, m2);
  return 0;
}
void _matmul_kernel_12x4(int r, double *A, int ldA, double *B, int ldB, double *C, int ldC) {
  /* Load micro-tile in the registers */
  __m256d C_0 = _mm256_loadu_pd(&ITEM(C, ldC, 0, 0)); 
  __m256d C_1 = _mm256_loadu_pd(&ITEM(C, ldC, 0, 1)); 
  __m256d C_2 = _mm256_loadu_pd(&ITEM(C, ldC, 0, 2)); 
  __m256d C_3 = _mm256_loadu_pd(&ITEM(C, ldC, 0, 3)); 
  __m256d C_4 = _mm256_loadu_pd(&ITEM(C, ldC, 4, 0)); 
  __m256d C_5 = _mm256_loadu_pd(&ITEM(C, ldC, 4, 1)); 
  __m256d C_6 = _mm256_loadu_pd(&ITEM(C, ldC, 4, 2)); 
  __m256d C_7 = _mm256_loadu_pd(&ITEM(C, ldC, 4, 3)); 
  __m256d C_8 = _mm256_loadu_pd(&ITEM(C, ldC, 8, 0)); 
  __m256d C_9 = _mm256_loadu_pd(&ITEM(C, ldC, 8, 1)); 
  __m256d C_10 = _mm256_loadu_pd(&ITEM(C, ldC, 8, 2)); 
  __m256d C_11 = _mm256_loadu_pd(&ITEM(C, ldC, 8, 3)); 
  for (int k = 0; k < r; k++) {
    /* Scalar from matrix B. Used to implement rank-1 multiplication */
    __m256d scalar; 
    __m256d A_col_0 = _mm256_loadu_pd(&ITEM(A, ldA, 0, k));
    __m256d A_col_4 = _mm256_loadu_pd(&ITEM(A, ldA, 4, k));
    __m256d A_col_8 = _mm256_loadu_pd(&ITEM(A, ldA, 8, k));

    scalar = _mm256_broadcast_sd(&ITEM(B, ldB, k, 0));
    C_0 = _mm256_fmadd_pd(A_col_0, scalar, C_0); 
    C_4 = _mm256_fmadd_pd(A_col_4, scalar, C_4); 
    C_8 = _mm256_fmadd_pd(A_col_8, scalar, C_8); 

    scalar = _mm256_broadcast_sd(&ITEM(B, ldB, k, 1));
    C_1 = _mm256_fmadd_pd(A_col_0, scalar, C_1); 
    C_5 = _mm256_fmadd_pd(A_col_4, scalar, C_5); 
    C_9 = _mm256_fmadd_pd(A_col_8, scalar, C_9); 

    scalar = _mm256_broadcast_sd(&ITEM(B, ldB, k, 2));
    C_2 = _mm256_fmadd_pd(A_col_0, scalar, C_2); 
    C_6 = _mm256_fmadd_pd(A_col_4, scalar, C_6); 
    C_10 = _mm256_fmadd_pd(A_col_8, scalar, C_10); 

    scalar = _mm256_broadcast_sd(&ITEM(B, ldB, k, 3));
    C_3 = _mm256_fmadd_pd(A_col_0, scalar, C_3); 
    C_7 = _mm256_fmadd_pd(A_col_4, scalar, C_7); 
    C_11 = _mm256_fmadd_pd(A_col_8, scalar, C_11); 
  }
  _mm256_storeu_pd(&ITEM(C, ldC, 0, 0), C_0);
  _mm256_storeu_pd(&ITEM(C, ldC, 0, 1), C_1);
  _mm256_storeu_pd(&ITEM(C, ldC, 0, 2), C_2);
  _mm256_storeu_pd(&ITEM(C, ldC, 0, 3), C_3);
  _mm256_storeu_pd(&ITEM(C, ldC, 4, 0), C_4);
  _mm256_storeu_pd(&ITEM(C, ldC, 4, 1), C_5);
  _mm256_storeu_pd(&ITEM(C, ldC, 4, 2), C_6);
  _mm256_storeu_pd(&ITEM(C, ldC, 4, 3), C_7);
  _mm256_storeu_pd(&ITEM(C, ldC, 8, 0), C_8);
  _mm256_storeu_pd(&ITEM(C, ldC, 8, 1), C_9);
  _mm256_storeu_pd(&ITEM(C, ldC, 8, 2), C_10);
  _mm256_storeu_pd(&ITEM(C, ldC, 8, 3), C_11);
}

void _matmul_kernel_4x4(int r, double *A, int ldA, double *B, int ldB, double *C, int ldC) {
  /* Load micro-tile in the registers */
  __m256d C_0 = _mm256_loadu_pd(&ITEM(C, ldC, 0, 0)); 
  __m256d C_1 = _mm256_loadu_pd(&ITEM(C, ldC, 0, 1)); 
  __m256d C_2 = _mm256_loadu_pd(&ITEM(C, ldC, 0, 2)); 
  __m256d C_3 = _mm256_loadu_pd(&ITEM(C, ldC, 0, 3)); 
  for (int k = 0; k < r; k++) {
    /* Scalar from matrix B. Used to implement rank-1 multiplication */
    __m256d scalar; 
    __m256d A_col = _mm256_loadu_pd(&ITEM(A, ldA, 0, k));

    scalar = _mm256_broadcast_sd(&ITEM(B, ldB, k, 0));
    C_0 = _mm256_fmadd_pd(A_col, scalar, C_0); 

    scalar = _mm256_broadcast_sd(&ITEM(B, ldB, k, 1));
    C_1 = _mm256_fmadd_pd(A_col, scalar, C_1); 

    scalar = _mm256_broadcast_sd(&ITEM(B, ldB, k, 2));
    C_2 = _mm256_fmadd_pd(A_col, scalar, C_2); 

    scalar = _mm256_broadcast_sd(&ITEM(B, ldB, k, 3));
    C_3 = _mm256_fmadd_pd(A_col, scalar, C_3); 
  }
  _mm256_storeu_pd(&ITEM(C, ldC, 0, 0), C_0);
  _mm256_storeu_pd(&ITEM(C, ldC, 0, 1), C_1);
  _mm256_storeu_pd(&ITEM(C, ldC, 0, 2), C_2);
  _mm256_storeu_pd(&ITEM(C, ldC, 0, 3), C_3);
}

void _matmul_block(int m, int n, int r, int ldA, int ldB, int ldC, double *res, double *block1, double *block2) {
  if (m == 4) {
    for (int j = 0; j < r; j++) {
      for (int k = 0; k < n; k++) {
          double b = ITEM(block2, ldB, k, j);
          ITEM(res, ldC, 0, j) += ITEM(block1, ldA, 0, k) * b;
          ITEM(res, ldC, 1, j) += ITEM(block1, ldA, 1, k) * b;
          ITEM(res, ldC, 2, j) += ITEM(block1, ldA, 2, k) * b;
          ITEM(res, ldC, 3, j) += ITEM(block1, ldA, 3, k) * b;
      }
    }
    return;
    
  }
  for (int j = 0; j < r; j++) {
    for (int k = 0; k < n; k++) {
      double b = ITEM(block2, ldB, k, j);
      for (int i = 0; i < m; i++) {
        ITEM(res, ldC, i, j) += ITEM(block1, ldA, i, k) * b;
      }
    }
  }
}

