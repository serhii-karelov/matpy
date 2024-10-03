#include <Python.h>
#include "matrix.h"
#define _mmul(suffix) PyObject* _pymatrix_matmul_ ## suffix(PyObject* self, PyObject* other) { \
  return _pymatrix_matmul_general(self, other, _matrix_matmul_ ## suffix); \
}
#define _mmul_meth(suffix) {"mul_" #suffix, (PyCFunction)  _pymatrix_matmul_ ## suffix, METH_O, ""}

PyObject* _pymatrix_matmul_1_naive_ijk(PyObject* self, PyObject* other);
PyObject* _pymatrix_matmul_2_naive_unrolled(PyObject* self, PyObject* other);

