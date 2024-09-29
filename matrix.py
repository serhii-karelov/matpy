import operator 
from timeit import timeit
from copy import deepcopy
from random import random 

class Matrix:
    def __init__(self, items, steal=False):
        rows = len(items)
        cols = len(items[0])
        if steal:
            self._items = items
        else:
            self._items = deepcopy(items)
        self.rows = rows
        self.cols = cols
        self._size = rows * cols
        self.shape = (self.rows, self.cols)

    @classmethod
    def fill(cls, rows, cols, value):
        items = [None] * rows
        for i in range(rows):
            items[i] = [0] * cols 
            for j in range(cols):
                items[i][j] = value
        return cls(items, steal=True)

    def _elementwise(self, other, op):
        if self.shape != other.shape:
            raise ValueError
        items = [None] * self.rows
        for i in range(self.rows):
            items[i] = [0] * self.cols 
            for j in range(self.cols):
                items[i][j] = op(self._items[i][j], other._items[i][j]) 
        return Matrix(items, steal=True)

    def __add__(self, other):
        return self._elementwise(other, operator.add)

    def __sub__(self, other):
        return self._elementwise(other, operator.sub)

    def __mul__(self, other):
        return self._elementwise(other, operator.mul)

    def __matmul__(self, other):
        if self.cols != other.rows:
            raise ValueError(f"Incompatible dimensions for matrix multiplication: "
                             f"{self.shape} and {other.shape}")
        new_items = [None] * self.rows
        for i in range(self.rows):
            new_items[i] = [0] * other.cols
            for j in range(self.cols):
                for k in range(other.cols):
                    new_items[i][k] += self._items[i][j] * other._items[j][k]
        return Matrix(new_items)

    def __repr__(self):
        return f'Matrix({self._items})'


m1 = Matrix.fill(1000, 1000, 800.1)
m2 = Matrix.fill(1000, 1000, 900.1)


#print(" matmul is done in %f s" % timeit(lambda: mv1 @ mv2, number=1))
print(" add is done in %f s" % timeit(lambda: m1 + m2, number=1))
print(" add is done in %f s" % timeit(lambda: m1 @ m2, number=1))

