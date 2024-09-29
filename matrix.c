#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <Python.h> /* PyMem_RawMalloc, PyMem_RawCalloc, PyMem_RawFree */
#include "matrix.h" /* typedef CMatrix */

#define MATRIX_ITEM(m, i, j) ((m)->items)[(i) + (j) * (m)->rows]

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
  if (!((result->rows == m1->rows)
        && (result->cols == m2->cols)
        && (m1->cols == m2->rows))) {
    return -1;
  }
  int m = m1->rows;
  int n = m2->cols;
  int r = m2->rows;

  for (int i = 0; i < m; i++) {
    for (int k = 0; k < r; k++) {
      for (int j = 0; j < n; j++) {
        MATRIX_ITEM(result, i, j) += MATRIX_ITEM(m1, i, k) * MATRIX_ITEM(m2, k, j);
      }
    }
  }
  return 0;
}

