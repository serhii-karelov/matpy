import pytest
import matpy
import matrix
import numpy as np

pytestmark = pytest.mark.parametrize("Matrix", [matpy.Matrix, matrix.Matrix])

def test_matrix_init(Matrix):
    items = [[1, 2, 3], [4, 5, 6]]
    m = Matrix(items) 
    assert items == m.as_list()

def test_matrix_fill(Matrix):
    expected = [[42.1, 42.1, 42.1], [42.1, 42.1, 42.1]]
    m = Matrix.fill(2, 3, 42.1) 
    assert m.as_list() == expected


def test_matpy_matrix_rand(Matrix):
    expected = [[0.8401877171547095, 0.39438292681909304, 0.7830992237586059],
                [0.7984400334760733, 0.9116473579367843, 0.19755136929338396]] 
    m = matpy.Matrix.rand(2, 3, seed=1) 
    assert m.as_list() == expected

def test_vanilla_matrix_rand(Matrix):
    expected = [ [0.13436424411240122, 0.8474337369372327, 0.763774618976614],
                [0.2550690257394217, 0.49543508709194095, 0.4494910647887381]]
    m = matrix.Matrix.rand(2, 3, seed=1) 
    assert m.as_list() == expected

def test_matrix_add(Matrix):
    m1 = Matrix([[1, 2, 3], [4, 5, 6]])
    m2 = Matrix([[10, 20, 30], [40, 50, 60]])
    result = m1 + m2
    expected = [[11, 22, 33], [44, 55, 66]]
    assert result.as_list() == expected

def test_matrix_sub(Matrix):
    m1 = Matrix([[1, 2, 3], [4, 5, 6]])
    m2 = Matrix([[10, 20, 30], [40, 50, 60]])
    result = m1 - m2
    expected = [[-9, -18, -27], [-36, -45, -54]]
    assert result.as_list() == expected

def test_matrix_mul(Matrix):
    m1 = Matrix([[1, 2, 3], [4, 5, 6]])
    m2 = Matrix([[10, 20, 30], [40, 50, 60]])
    result = m1 * m2
    expected = [[10, 40, 90], [160, 250, 360]]
    assert result.as_list() == expected

def test_matrix_matmul(Matrix):
    m1 = Matrix([[1, 2, 3], [4, 5, 6]])
    m2 = Matrix(([[10, 40], [20, 50], [30, 60]]))
    result = m1 @ m2
    expected = [[140, 320], [320, 770]]
    assert result.as_list() == expected

def test_large_matrix_matmul(Matrix):
    np.random.seed(42)
    a = np.random.rand(127, 120) 
    b = np.random.rand(120, 137) 
    c = a @ b
    print(a.tolist())
    m1 = Matrix(a.tolist())
    m2 = Matrix(b.tolist())
    result = m1 @ m2
    result = result.as_list()
    for row in range(len(result)):
        assert pytest.approx(result[row]) == c[row].tolist()

# TEST MATMUL VARIATIONS
_VARIATION_METHODS = [
    'mul_1_ikj',
    'mul_2_kji',
    'mul_3_kji_unrolled',
    'mul_4_register_blocked',
]
def test_matmul_variations_small(Matrix):
    m1 = matpy.Matrix([[1, 2, 3], [4, 5, 6]])
    m2 = matpy.Matrix(([[10, 40], [20, 50], [30, 60]]))
    expected = [[140, 320], [320, 770]]
    for meth in _VARIATION_METHODS:
        result = getattr(m1, meth)(m2)
        assert result.as_list() == expected, f"Multipling via {meth}"

def test_matmul_variations_large(Matrix):
    np.random.seed(42)
    a = np.random.rand(127, 131) 
    b = np.random.rand(131, 199) 
    c = a @ b
    m1 = matpy.Matrix(a.tolist())
    m2 = matpy.Matrix(b.tolist())
    for meth in _VARIATION_METHODS:
        result = getattr(m1, meth)(m2)
        result = result.as_list()
        for row in range(len(result)):
            assert pytest.approx(result[row]) == c[row].tolist(), f"Multipling via {meth}"

