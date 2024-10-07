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

def report_memory_vanilla():
    print("### Memory allocations of pure-Python implementation ###")
    tracemalloc.start()
    m1 = matrix.Matrix.rand(2048 * 2, 2048 * 2)
    snap = tracemalloc.take_snapshot()
    tracemalloc.stop()
    for stat in snap.statistics("lineno"):
        print(stat)


def report_memory_matpy():
    print("### Memory allocations of matpy implementation ###")
    tracemalloc.start()
    m1 = matpy.Matrix.rand(2048 * 2, 2048 * 2)
    snap = tracemalloc.take_snapshot()
    tracemalloc.stop()
    for stat in snap.statistics("lineno"):
        print(stat)


if __name__ == "__main__":
    report_memory_matpy()
    report_memory_vanilla()
    """
    vmmul, vmadd = measure_vanilla()
    mmul, madd = measure_matpy()

    print(f"Pyre-python matmul: {vmmul:.2f}s \t add: {vmadd:.3f}s")
    print(f"matpy matmul matmul: {mmul:.2f}s \t add: {madd:.3f}s")
    print(f"Speed up for matmul is {(vmmul / mmul):.1f}x and for add is {(vmadd / madd):.1f}x") 
    """


