/*
 *
 * Allocating less memory in GPU than required
 * Matrix multiplication
 * Vector_addition:unspecified launch failure
 * Matrix multiplication:invalid argument while copying
 * 
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__
void matrix_mul(float *matrix_1, float *matrix_2, float *result, int dimension) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float Pvalue = 0;
	float el1, el2;

    for (int i=0; i <= dimension; ++i) {
        el1 = matrix_1[ty*dimension + i];
        el2 = matrix_2[i*dimension + tx];
        Pvalue += el1*el2;
    }
	
    result[ty*dimension + tx] = Pvalue;
}

int main() {
    int sq_dimension = 256;
    int size = sq_dimension * sq_dimension * sizeof(float);
	float *h_matrix_1, *h_matrix_2, *h_result;
    float *d_matrix_1, *d_matrix_2, *d_result;

	h_matrix_1 = (float*)malloc(size);
	h_matrix_2 = (float*)malloc(size);
	h_result = (float*)malloc(size);
	
    for (int i = 0; i < sq_dimension; i++) {
        h_matrix_1[i] = i + j + 0.5;
        h_matrix_2[i] = i + j + 0.8;
        h_result[i] = 0;
        }
    }		
		
    cudaMalloc(&d_matrix_1, size); 
	cudaMalloc(&d_matrix_2, size);
	cudaMalloc(&d_result, size-sizeof(float));
	
	cudaMemcpy(d_matrix_1, h_matrix_1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrix_2, h_matrix_2, size, cudaMemcpyHostToDevice);
	gpuErrchk(cudaMemcpy(d_result, h_result, size, cudaMemcpyHostToDevice));

    dim3 dimBlock(sq_dimension, sq_dimension);
    dim3 dimGrid(1,1);
    matrix_mul<<<dimGrid, dimBlock, dimBlock.x * dimBlock.x * sizeof(float)>>>(d_matrix_1, d_matrix_2, d_result, sq_dimension);

    cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
	
    cudaFree(d_matrix_1);
    cudaFree(d_matrix_2);
    cudaFree(d_result);
	
	free(h_matrix_1);
	free(h_matrix_2);
	free(h_result);
}

