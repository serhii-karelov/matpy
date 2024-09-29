#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <Python.h> /* PyMem_RawMalloc, PyMem_RawCalloc, PyMem_RawFree */
#include "matrix.h" /* typedef Matrix */

#define MATRIX_GET(m, i, j) ((m)->items)[(i) + (j) * (m)->rows]
#define MATRIX_SET(m, i, j, x) ((m)->items)[(i) + (j) * (m)->rows] = x
#define MATRIX_ADDSET(m, i, j, x) ((m)->items)[(i) + (j) * (m)->rows] += x

/* 
 * TODO
 *  - row-major order
 *  - kernalize => register blocking using SIMD
 * */
 
int matrix_alloc(Matrix** m, int rows, int cols) {
  (*m) = (Matrix *) PyMem_RawMalloc(sizeof(Matrix));
  if ((*m) == NULL) {
    return -1;
  }
  (*m)->rows = rows;
  (*m)->cols = cols;
  (*m)->items = (double*) PyMem_RawCalloc(rows * cols, sizeof(double));
  return 0;
}

void matrix_dealloc(Matrix* m) {
  if (m == NULL) {
    return;
  }
  if (m->items != NULL) {
    PyMem_RawFree(m->items);    // free memory holding pointers to rows
  }
  PyMem_RawFree(m);
}

void matrix_fill(Matrix* m, double x) {
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      MATRIX_SET(m, i, j, x);
    } 
  }
}

void matrix_fill_rand(Matrix* m, int seed) {
  srand(seed);
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      MATRIX_SET(m, i, j, ((double) random()) / 2147483647);
    } 
  }
}

void matrix_set(Matrix* m, int i, int j, double x) {
  (m->items)[i + j * m->rows] = x;
}

double matrix_get(Matrix* m, int i, int j) {
  return (m->items)[i + j * m->rows];
}

static int check_dimensions(Matrix* m1, Matrix* m2, Matrix* m3) {
  return ((m1->rows == m2->rows)
       && (m1->rows == m3->rows)
       && (m1->cols == m2->cols)
       && (m1->cols == m3->cols));
}

static double matrix_elementwise(Matrix* result, Matrix* m1, Matrix* m2, double (*op)(double, double)) {
  if (!check_dimensions(result, m1, m2)) {
    return -1;
  }
  double res;

  for (int i = 0; i < m1->rows; i++) {
    for (int j = 0; j < m2->cols; j++) {
      res = (*op)(MATRIX_GET(m1, i, j), MATRIX_GET(m2, i, j));
      MATRIX_SET(result, i, j, res);
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

int matrix_add(Matrix* result, Matrix* m1, Matrix* m2) {
  return matrix_elementwise(result, m1, m2, op_add);
}
  
int matrix_sub(Matrix* result, Matrix* m1, Matrix* m2) {
  return matrix_elementwise(result, m1, m2, op_sub);
}

int matrix_mul(Matrix* result, Matrix* m1, Matrix* m2) {
  return matrix_elementwise(result, m1, m2, op_mul);
}

int matrix_matmul(Matrix* result, Matrix* m1, Matrix* m2) {
  if (!((result->rows == m1->rows)
        && (result->cols == m2->cols)
        && (m1->cols == m2->rows))) {
    return -1;
  }
  double a, b;
  int k;
  for (int i = 0; i < m1->rows; i++) {
    for (int j = 0; j < m1->cols; j++) {
      a = MATRIX_GET(m1, i, j);
      for (k = 0; k < m2->cols; k++) {
        b = MATRIX_GET(m2, j, k);
        MATRIX_ADDSET(result, i, k, a * b);
      }
    }
  }
  return 0;
}

int matrix_matmul_kji(Matrix* result, Matrix* m1, Matrix* m2) {
  if (!((result->rows == m1->rows)
        && (result->cols == m2->cols)
        && (m1->cols == m2->rows))) {
    return -1;
  }
  double a, b;
  int k;
  for (k = 0; k < m2->cols; k++) {
    for (int j = 0; j < m1->cols; j++) {
      b = MATRIX_GET(m2, j, k);
      for (int i = 0; i < m1->rows; i++) {
        a = MATRIX_GET(m1, i, j);
        MATRIX_ADDSET(result, i, k, a * b);
      }
    }
  }
  return 0;
}

int matrix_matmul_unrolled(Matrix* result, Matrix* m1, Matrix* m2) {
  if (!((result->rows == m1->rows)
        && (result->cols == m2->cols)
        && (m1->cols == m2->rows))) {
    return -1;
  }
  double a, b[1];
  int k;
  for (int i = 0; i < m1->rows; i++) {
    for (int j = 0; j < m1->cols; j++) {
      a = MATRIX_GET(m1, i, j);
      for (k = 0; k < m2->cols - 8; k +=8) {
        b[0] = MATRIX_GET(m2, j, k);
        b[1] = MATRIX_GET(m2, j, k + 1);
        b[2] = MATRIX_GET(m2, j, k + 2);
        b[3] = MATRIX_GET(m2, j, k + 3);
        b[4] = MATRIX_GET(m2, j, k + 4);
        b[5] = MATRIX_GET(m2, j, k + 5);
        b[6] = MATRIX_GET(m2, j, k + 6);
        b[7] = MATRIX_GET(m2, j, k + 7);
        MATRIX_ADDSET(result, i, k,     a * b[0]);
        MATRIX_ADDSET(result, i, k + 1, a * b[1]);
        MATRIX_ADDSET(result, i, k + 2, a * b[2]);
        MATRIX_ADDSET(result, i, k + 3, a * b[3]);
        MATRIX_ADDSET(result, i, k + 4, a * b[4]);
        MATRIX_ADDSET(result, i, k + 5, a * b[5]);
        MATRIX_ADDSET(result, i, k + 6, a * b[6]);
        MATRIX_ADDSET(result, i, k + 7, a * b[7]);
      }
      for (; k < m2->cols; k++) {
        b[0] = MATRIX_GET(m2, j, k);
        MATRIX_ADDSET(result, i, k, a * b[0]);
      } 
    }
  }
  return 0;
}

// up to 50-55 probably can exploit locality
#define block_size 32

#define min(a, b) a < b ? a : b

int matrix_matmul_blocked(Matrix* result, Matrix* m1, Matrix* m2) {
  if (!((result->rows == m1->rows)
        && (result->cols == m2->cols)
        && (m1->cols == m2->rows))) {
    return -1;
  }
  int blocks_m = m1->rows / block_size; 
  int blocks_n = m1->cols / block_size; 
  int blocks_r = m2->cols / block_size; 

  int a, b;
  int i_limit;
  int j_limit;
  int k_limit;
  for (int block_i = 0; block_i <= blocks_m; block_i++) {
    for (int block_k = 0; block_k <= blocks_r; block_k++) {
      for (int block_j = 0; block_j <= blocks_n; block_j++) {
        i_limit = min(m1->rows, (block_i + 1) * block_size);
        j_limit = min(m1->cols, (block_j + 1) * block_size);
        k_limit = min(m2->cols, (block_k + 1) * block_size);
        for (int i = block_i * block_size; i < i_limit; i++) {
          for (int j = block_j * block_size; j < j_limit; j++) {
            a = MATRIX_GET(m1, i, j);
            for (int k = block_k * block_size; k < k_limit; k++) {
              b = MATRIX_GET(m2, j, k);
              MATRIX_ADDSET(result, i, k, a * b);
            }
          }
        }
      }
    }
  }
}

