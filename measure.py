from timeit import timeit

from matpy import Matrix
import numpy as np


def measure_matpy():
    m1 = Matrix.fill(1024, 1024, 800.1)
    m2 = Matrix.fill(1024, 1024, 900.1)

    print("C matmul is done in %f s" % timeit(lambda: m1 @ m2, number=1))
    print("C add is done in %f s" % timeit(lambda: m1 + m2, number=1))

def measure_numpy():
    a1 = np.full((1024, 1024), 800.1)
    a2 = np.full((1024, 1024), 900.1)

    print("numpy matmul is done in %f s" % timeit(lambda: a1 @ a2, number=1))
    print("numpy add is done in %f s" % timeit(lambda: a1 + a2, number=1))

if __name__ == "__main__":
    measure_matpy()
    measure_numpy()

