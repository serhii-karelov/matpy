import pytest

from matpy import Matrix

def test_matrix_init():
    items = [[1, 2, 3], [4, 5, 6]]
    m = Matrix(items) 
    assert items == m.as_list()

def test_matrix_fill():
    expected = [[42.1, 42.1, 42.1], [42.1, 42.1, 42.1]]
    m = Matrix.fill(2, 3, 42.1) 
    assert m.as_list() == expected


def test_matrix_rand():
    expected = [[0.8401877171547095, 0.39438292681909304, 0.7830992237586059],
                [0.7984400334760733, 0.9116473579367843, 0.19755136929338396]] 
    m = Matrix.rand(2, 3, seed=1) 
    assert m.as_list() == expected

def test_matrix_add():
    m1 = Matrix([[1, 2, 3], [4, 5, 6]])
    m2 = Matrix([[10, 20, 30], [40, 50, 60]])
    result = m1 + m2
    expected = [[11, 22, 33], [44, 55, 66]]
    assert result.as_list() == expected

def test_matrix_sub():
    m1 = Matrix([[1, 2, 3], [4, 5, 6]])
    m2 = Matrix([[10, 20, 30], [40, 50, 60]])
    result = m1 - m2
    expected = [[-9, -18, -27], [-36, -45, -54]]
    assert result.as_list() == expected

def test_matrix_mul():
    m1 = Matrix([[1, 2, 3], [4, 5, 6]])
    m2 = Matrix([[10, 20, 30], [40, 50, 60]])
    result = m1 * m2
    expected = [[10, 40, 90], [160, 250, 360]]
    assert result.as_list() == expected

def test_matrix_matmul():
    m1 = Matrix([[1, 2, 3], [4, 5, 6]])
    m2 = Matrix(([[10, 40], [20, 50], [30, 60]]))
    result = m1 @ m2
    expected = [[140, 320], [320, 770]]
    assert result.as_list() == expected

