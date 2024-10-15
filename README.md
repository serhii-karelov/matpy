## `matpy` library
This is educational numerical library, where I have implemented fast matrix
multiplication based on the ideas from the BLIS project (
[Paper 3](https://www.cs.utexas.edu/~flame/pubs/blis3_ipdps14.pdf),
[Paper 1](https://www.cs.utexas.edu/~flame/pubs/blis1_toms_rev3.pdf), https://github.com/flame/blis). 

The library provides built-in `Matrix` type, that can be imported and used in Python. 
The type itself and underlying algorithms are implemented in C. It is 1400x faster than pure-python implementation. 

The library written and published as part of series of articles I am writing right now where I teach computer science topics (coming this month). 

### Ideas implemented in this project
- Instruction level paralellism (SIMD).
- Thread parallelism.
- Blocking for for memory hierarchy
    - blocking for registers with 12x4 kernel
    - blocking for L1, L2, L3 caches
    - streaming of data from caches

### Ideas not implemented (although straightforward to implement)
- Data packing

### Experiments conducted (and other tricks of the trade)
- Loop unrolling gives additional speed up
- The right order of loops plus `-O3` optimizations give good level of performance 
virtually for free
- Kernel of size 4x4 gives slightly poorer performance than 12x4

### Pure Python implementation
For comparison, I provide pure-python implementation in `matrix.py`. It is serves as
a base line. 
The final result gives 1400x faster that the base line.

#### Measure the performance of C vs Python implementation 

```shell
linux> python measure.py
pure-python is done: 480.51s     add: 0.387s
matpy matmul is done: 0.34s      add: 0.056s
```

### Run tests
pytest test.py

### Build from source
On 4 core system:
```
OMP_NUM_THREADS=4 pip install . --no-build-isolation
```
