from timeit import timeit

import numpy as np

import matpy
import matrix


def measure_vanilla():
    m1 = matrix.Matrix.fill(1024, 1024, 800.1)
    m2 = matrix.Matrix.fill(1024, 1024, 900.1)

    print("Vanilla matmul is done in %f s" % timeit(lambda: m1 @ m2, number=1))
    print("Vanilla add is done in %f s" % timeit(lambda: m1 + m2, number=1))

def measure_matpy():
    m1 = matpy.Matrix.fill(1024, 1024, 800.1)
    m2 = matpy.Matrix.fill(1024, 1024, 900.1)

    print("C matmul is done in %f s" % timeit(lambda: m1 @ m2, number=1))
    print("C add is done in %f s" % timeit(lambda: m1 + m2, number=1))



def measure_numpy():
    a1 = np.full((1024, 1024), 800.1)
    a2 = np.full((1024, 1024), 900.1)

    print("numpy matmul is done in %f s" % timeit(lambda: a1 @ a2, number=1))
    print("numpy add is done in %f s" % timeit(lambda: a1 + a2, number=1))

# VARIATIONS START
_VARIATION_METHODS = [
    'mul_1_ikj',
    #'mul_2_kji',
    # 'mul_2_kji',
    #'mul_3_kji_unrolled',
    'mul_4_register_blocked',
    'mul_5_loops',
]

def _measure_variations():
    m1 = matpy.Matrix.fill(1024, 1024, 800.1)
    m2 = matpy.Matrix.fill(1024, 1024, 900.1)
    for meth in _VARIATION_METHODS:
        print(f"measuring {meth}")
        print(f"C matmul via {meth} is done in %f s" % timeit(lambda: getattr(m1, meth)(m2), number=1))
# VARIATIONS END

if __name__ == "__main__":
    # measure_vanilla()
    measure_matpy()
    _measure_variations()
    measure_numpy()

