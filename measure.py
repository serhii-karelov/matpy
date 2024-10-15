import tracemalloc

from timeit import timeit

import matpy
import matrix


def measure_vanilla(times=1):
    m1 = matrix.Matrix.rand(1200, 1200)
    m2 = matrix.Matrix.rand(1200, 1200)
    time_mmul = (timeit(lambda: m1 @ m2, number=times) / times)
    time_add = (timeit(lambda: m1 + m2, number=times) / times) 
    return  time_mmul, time_add

def measure_matpy(times=1):
    m1 = matpy.Matrix.rand(1200, 1200)
    m2 = matpy.Matrix.rand(1200, 1200)
    time_mmul = (timeit(lambda: m1 @ m2, number=times) / times)
    time_add = (timeit(lambda: m1 + m2, number=times) / times) 
    return  time_mmul, time_add

if __name__ == "__main__":
    mmul, madd = measure_vanilla(times=1)
    mmul, madd = measure_matpy(times=10)

    print(f"matpy matmul matmul: {(mmul):.2f}s \t add: {madd:.3f}s")


