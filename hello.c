#include <stdio.h>
#include <omp.h>

void main(int argc, char** argv) {
#pragma omp parallel 
  printf("Hello world!\n");

}

