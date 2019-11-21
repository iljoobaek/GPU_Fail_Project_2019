/*
 *
 * Datatype overflow from GPU
 * array sum
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
void array_sum(int n, int *e_x, int *e_z) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    //Inserted Error: e_z out of bounds
        for (id = 0; id < n; id++) {
            *e_z += e_x[id] * 4;
            //Overflow error check
            if (*e_z < 0) {
               printf("GPU ERROR %d %d\n", blockIdx.x, threadIdx.x);
               return;
            }
        }
}

int main() {
    int no_el = 1048576;
    int block_size = 512;
    int grid_size = (no_el/block_size) + 1;    //ceil doesn't give correct grid size

    int *h_x, *h_z;
    h_x = (int*)malloc(no_el*sizeof(int));
    h_z = (int*)malloc(sizeof(int));

    *h_z = 0;
    for (int i = 0; i < no_el; i++) {
            h_x[i] = i + 200;     //sum = 524288(0+1048575)=already out of bounds
    }

    int *d_x, *d_z;
    cudaMalloc(&d_x, no_el*sizeof(int));
    cudaMalloc(&d_z, sizeof(int));

    cudaMemcpy(d_x, h_x, no_el*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(block_size);
    dim3 grid(grid_size);

    array_sum<<<grid, block>>>(no_el, d_x, d_z);

    gpuErrchk(cudaMemcpy(h_z, d_z, sizeof(int), cudaMemcpyDeviceToHost));

    printf("result = %d\n", *h_z);
    cudaFree(d_x);
    cudaFree(d_z);

    free(h_x);
    free(h_z);

}
