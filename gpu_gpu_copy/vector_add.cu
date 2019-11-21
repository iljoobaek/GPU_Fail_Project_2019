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
void vector_addition(double n, int *e_x, int *e_y, int *e_z, int *e_z_copy, int *e_k_copy) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) { //for (id = 0; id <= n; id++) {
        e_z[id] = (2*(e_x[id] + e_y[id]));
        e_k_copy[id] = 1;
        //if (id == 1000)
        //    e_k_copy[id] = 0;
	    e_z_copy[id] = e_z[id];
        if (id == n-1)
            e_z_copy[id+1000] = e_z[id];
    }
}

int check_results(int *result, int n) {
    int i = 0;
    for (i = 0; i < n; i++) {
	if (result[i] != (2 * (i + i + 2))) {
	    printf("result fails at index %d\n", i);
	    return 0;
	}
    }
    printf("result is correct\n");
    return 1;
}

int check_results2(int *result, int n) {
    int i = 0;
    for (i = 0; i < n; i++) {
	if (result[i] != 1) {
	    printf("result fails at index %d !!\n", i);
	    return 0;
	}
    }
    printf("result is correct !!\n");
    return 1;
}

int main() {
    int no_el = 268435456;
    int block_size = 1024;
    int grid_size = (no_el/block_size) + 1;                    //ceil doesn't give correct grid size

    int *h_x, *d_x, *h_y, *d_y, *h_z, *d_z;
    int *h_z_copy, *d_z_copy;
    int *h_k_copy, *d_k_copy;

    h_x = (int*)malloc(no_el*sizeof(int));
    h_y = (int*)malloc(no_el*sizeof(int));
    h_z = (int*)malloc(no_el*sizeof(int));
    h_z_copy = (int*)malloc((no_el-1000)*sizeof(int));
    h_k_copy = (int*)malloc(no_el*sizeof(int));

    int i = 0;
    for (i = 0; i < no_el; i++) {
        h_x[i] = i;
        h_y[i] = i + 2;
        h_z[i] = 0;
    }
   // printf("allocated x, y, z\n");

    for (i = 0; i < (no_el - 1000); i++) {
	    h_z_copy[i] = 0;
    }

    for (i = 0; i < no_el; i++) {
	    h_k_copy[i] = 0;
    }
   // printf("filled up h_z_copy array\n");

    cudaMalloc(&d_x, no_el*sizeof(int));
    cudaMalloc(&d_y, no_el*sizeof(int));
    cudaMalloc(&d_z, no_el*sizeof(int));
    cudaMalloc(&d_z_copy, (no_el-1000)*sizeof(int));
    cudaMalloc(&d_k_copy, no_el*sizeof(int));
   // printf("cuda malloc succeeded\n");

    cudaMemcpy(d_x, h_x, no_el*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, no_el*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, no_el*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z_copy, h_z_copy, (no_el-1000)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_copy, h_k_copy, no_el*sizeof(int), cudaMemcpyHostToDevice);
   // printf("cuda memcpy succeeded\n");

    while(1) {
        dim3 block(block_size);
        dim3 grid(grid_size);

        vector_addition<<<grid, block>>>(no_el, d_x, d_y, d_z, d_z_copy, d_k_copy);
        printf("kernel launch succeeded\n");
        gpuErrchk(cudaMemcpy(h_z, d_z, no_el*sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_k_copy, d_k_copy, no_el*sizeof(int), cudaMemcpyDeviceToHost));
        printf("copied result from gpu to cpu successfully\n");
        int valid = check_results(h_z, no_el);
    	if (valid == 0) {
    	    printf("wrong computation result\n");
    	    //break;
    	}

        int valid2 = check_results2(h_k_copy, no_el);
    	if (valid2 == 0) {
    	    printf("wrong computation result !!\n");
    	    //break;
    	}
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_z_copy);

    free(h_x);
    free(h_y);
    free(h_z);
    free(h_z_copy);

}
