#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stddef.h>
#include "matrix.h"

/* TODO: Rename Matrix to CMatrix
 *
 * */
static PyObject* MatrixError;
static PyTypeObject MatrixType;

typedef struct {
  PyObject_HEAD
  Matrix* matrix;
  PyObject* shape;
} MatrixObject; 

void pymatrix_dealloc(PyObject* self) {
  // printf("Deallocating matrix...\n");
  MatrixObject* m = (MatrixObject*) self;
  Py_XDECREF(m->shape);
  matrix_dealloc(m->matrix);
  Py_TYPE(m)->tp_free(self);
}

/* Initializes a newly allocated Python object with zeroed matrix
 * and `shape` tuple that holds matrix dimensions.
 * */
static int _pymatrix_init_empty(PyObject* mat, int rows, int cols) {
  MatrixObject* m = (MatrixObject* ) mat;
  Matrix* matrix;
  if (matrix_alloc(&matrix, rows, cols)) {
    PyErr_NoMemory();
    return -1;
  }
  m->matrix = matrix;
  m->shape = Py_BuildValue("ii", rows, cols);
  if (m->shape == NULL) {
    matrix_dealloc(m->matrix);
    return -1;
  }
  return 0;
}

/* Constructs a partially-initialized MatrixObject.
 * Has two steps: 
 *   1. Allocate new object with type->tp_alloc 
 *   2. Initialize with empty matrix with _pymatrix_init_empty 
 * */
static PyObject* _pymatrix_construct(PyTypeObject* type, int rows, int cols) {
  PyObject* m;
  m = type->tp_alloc(type, 0);
  if (m == NULL) {
    return NULL;
  }
  if (_pymatrix_init_empty(m, rows, cols)) {
    return NULL;
  }
  return m;
}
static PyObject* pymatrix_elementwise(PyObject* self, PyObject* other, int (*op)(Matrix*, Matrix*, Matrix*)) {
  Matrix* m1 = ((MatrixObject*) self)->matrix;
  Matrix* m2 = ((MatrixObject*) other)->matrix;
  if (!(m1->rows == m2->rows) && (m1->cols == m2->cols)) {
    PyErr_SetString(MatrixError, "Dimensions mismatch");
    return NULL;
  }
  PyObject* result = _pymatrix_construct(&MatrixType, m1->rows, m1->cols);
  if (result == NULL) {
    return NULL;
  }
  (*op)(((MatrixObject*) result)->matrix, m1, m2);
  return result;
}

PyObject* pymatrix_add(PyObject* self, PyObject* other) {
  return pymatrix_elementwise(self, other, matrix_add);
}

PyObject* pymatrix_sub(PyObject* self, PyObject* other) {
  return pymatrix_elementwise(self, other, matrix_sub);
}

PyObject* pymatrix_mul(PyObject* self, PyObject* other) {
  return pymatrix_elementwise(self, other, matrix_mul);
}

PyObject* pymatrix_matmul(PyObject* self, PyObject* other) {
  Matrix* m1 = ((MatrixObject*) self)->matrix;
  Matrix* m2 = ((MatrixObject*) other)->matrix;
  if (!(m1->cols == m2->rows)) {
    PyErr_SetString(MatrixError, "Dimensions mismatch");
    return NULL;
  }
  PyObject* result = _pymatrix_construct(&MatrixType, m1->rows, m2->cols);
  if (result == NULL) {
    return NULL;
  }
  matrix_matmul(((MatrixObject*) result)->matrix, m1, m2);
  return result;
}

PyObject* pymatrix_fill(PyTypeObject* type, PyObject* args) {
  int rows, cols;
  double x;
  if (!PyArg_ParseTuple(args, "iid", &rows, &cols, &x)) {
    return NULL;
  }
  MatrixObject* m = (MatrixObject*) _pymatrix_construct(type, rows, cols);
  matrix_fill(m->matrix, x);
  return (PyObject*) m;
}

PyObject* pymatrix_rand(PyTypeObject* type, PyObject* args, PyObject* kw) {
  int rows, cols;
  int seed = 42;
  char* keys[] = {"rows", "cols", "seed", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kw, "ii|$i", keys, &rows, &cols, &seed)) {
    return NULL;
  }

  MatrixObject* m = (MatrixObject*) _pymatrix_construct(type, rows, cols);
  matrix_fill_rand(m->matrix, seed);
  return (PyObject*) m;
}

PyObject* pymatrix_as_list(PyObject* self, PyObject* Py_UNUSED(ignored)) {
  Matrix* matrix = ((MatrixObject*) self)->matrix;
  PyObject* rows = PyList_New(matrix->rows);
  PyObject* col;
  for (int i = 0; i < matrix->rows; i++) {
    col = PyList_New(matrix->cols);
    for (int j = 0; j < matrix->cols; j++) {
      PyList_SET_ITEM(col, j, PyFloat_FromDouble(matrix_get(matrix, i, j)));
    }
    PyList_SET_ITEM(rows, i, col);
  }
  return rows;
}

static int pymatrix_init(PyObject *self, PyObject* args, PyObject* kwagrs) {
  PyObject* items_list;
  int rows, cols;
  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &items_list)) {
    PyErr_SetString(MatrixError,
                    "Matrix can be initialized only from a 2D list."
                    "You can also use alternative constructors, "
                    "Matrix.fill(...) or Matrix.rand(...).");
    return -1;
  }
  rows = PyList_Size(items_list);
  cols = PyList_Size(PyList_GetItem(items_list, 0));
  
  _pymatrix_init_empty(self, rows, cols);

  MatrixObject* m = (MatrixObject *) self;
  PyObject* row; 
  double item;
  for (int i = 0; i < rows; i++) {
    row = PyList_GetItem(items_list, i); 
    for (int j = 0; j < cols; j++) {
      item = PyLong_AsLong(PyList_GetItem(row, j));
      matrix_set(m->matrix, i, j, item);
    }
  }
  return 0; 
}

static PyMemberDef pymatrix_members[] = {
  {"shape", Py_T_OBJECT_EX, offsetof(MatrixObject, shape),
    Py_READONLY, "Matrix dimensions"},
  {NULL}, /* Terminator */
};

static PyMethodDef pymatrix_methods[] = {
  {"as_list", (PyCFunction) pymatrix_as_list,
    METH_NOARGS, "Return matrix items as list"},
  {"fill", (PyCFunction) pymatrix_fill,
    METH_CLASS | METH_VARARGS, "Create matrix filled with the specified value"},
  {"rand", (PyCFunction) pymatrix_rand,
    METH_CLASS | METH_VARARGS | METH_KEYWORDS, "Create matrix filled with random values"},
  {NULL}, /* Terminator */
};

static PyNumberMethods pymatrix_as_number = {
    .nb_add = (binaryfunc) pymatrix_add,
    .nb_subtract = (binaryfunc) pymatrix_sub,
    .nb_multiply = (binaryfunc) pymatrix_mul,
    .nb_matrix_multiply = (binaryfunc) pymatrix_matmul,
};

static PyTypeObject MatrixType = {
  .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "matpy.Matrix",
  .tp_basicsize = sizeof(MatrixObject),
  .tp_itemsize = 0,
  .tp_new = PyType_GenericNew,
  .tp_init = (initproc) pymatrix_init,
  .tp_dealloc = (destructor) pymatrix_dealloc,
  .tp_members = pymatrix_members,
  .tp_methods = pymatrix_methods,
  .tp_as_number = &pymatrix_as_number
};

static struct PyModuleDef matpymodule = {
  PyModuleDef_HEAD_INIT,
  "matpy",
  "A matrix library",
  -1,
};

PyMODINIT_FUNC PyInit_matpy(void) {
  printf("Matpy is imported!\n");

  if (PyType_Ready(&MatrixType) < 0) { /* finish initialization of the type */
    return NULL;
  }
  PyObject* module = PyModule_Create(&matpymodule);
  if (module == NULL) {
    return NULL;
  }

  Py_INCREF(&MatrixType); 
  if (PyModule_AddObject(module, "Matrix", (PyObject* ) &MatrixType) < 0) {
    Py_DECREF(&MatrixType);
    Py_DECREF(module);
    return NULL;
  }

  MatrixError = PyErr_NewException("matpy.error", NULL, NULL); 
  Py_XINCREF(MatrixError); 

  if (PyModule_AddObject(module, "MatrixError", MatrixError)) { 
    Py_XDECREF(MatrixError);
    Py_CLEAR(MatrixError);
    Py_DECREF(module); 
    return NULL;
  }

  return module;
}

