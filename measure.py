import tracemalloc

from timeit import timeit

import matpy
import matrix


def measure_vanilla(times=1):
    m1 = matrix.Matrix.fill(1024, 1024, 800.1)
    m2 = matrix.Matrix.fill(1024, 1024, 900.1)
    time_mmul = (timeit(lambda: m1 @ m2, number=times) / times)
    time_add = (timeit(lambda: m1 + m2, number=times) / times) 
    return  time_mmul, time_add

def measure_matpy(times=1):
    m1 = matpy.Matrix.fill(1024, 1024, 800.1)
    m2 = matpy.Matrix.fill(1024, 1024, 900.1)
    time_mmul = (timeit(lambda: m1 @ m2, number=times) / times)
    time_add = (timeit(lambda: m1 + m2, number=times) / times) 
    return  time_mmul, time_add

if __name__ == "__main__":
    mmul, madd = measure_matpy()

    print(f"matpy matmul matmul: {(mmul):.2f}s \t add: {madd:.3f}s")


