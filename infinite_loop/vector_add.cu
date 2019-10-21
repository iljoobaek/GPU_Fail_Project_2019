/*
 *
 * Accessing out of bound memory from GPU
 * Vector addition
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__
void vector_addition(int n, int *e_x, int *e_y, int *e_z) {
    //int id = blockIdx.x * blockDim.x + threadIdx.x;
    //Inserted Error: while(1) loop
    while(1) {
    }
}

int main() {
    int no_el = 1048576;
    int block_size = 512;
    int grid_size = (no_el/block_size) + 1;                    //ceil doesn't give correct grid size

    int *h_x, *d_x, *h_y, *d_y, *h_z, *d_z;

    h_x = (int*)malloc(no_el*sizeof(int));
    h_y = (int*)malloc(no_el*sizeof(int));
    h_z = (int*)malloc(no_el*sizeof(int));

    for (int i = 0; i < no_el; i++) {
        h_x[i] = i;
        h_y[i] = i + 2;
        h_z[i] = 0;
    }

    cudaMalloc(&d_x, no_el*sizeof(int));
    cudaMalloc(&d_y, no_el*sizeof(int));
    cudaMalloc(&d_z, no_el*sizeof(int));

    cudaMemcpy(d_x, h_x, no_el*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, no_el*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, no_el*sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(block_size);
    dim3 grid(grid_size);

    vector_addition<<<grid, block>>>(no_el, d_x, d_y, d_z);

    gpuErrchk(cudaMemcpy(h_z, d_z, no_el*sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    free(h_x);
    free(h_y);
    free(h_z);
}
