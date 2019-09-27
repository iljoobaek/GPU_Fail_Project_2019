#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <iostream>

using namespace std;

// device memory
float* dev_A;
float* dev_B;
float* dev_out;

// host memory
float* matrix1;
float* matrix2;
float* outBuffer;


inline void CHECKCUDA(cudaError_t e)
{
	if (e != cudaSuccess)
	{
		cerr<<"CUDA Error:"<< cudaGetErrorString(e) << endl;
		exit(1);
	}
}

__device__
void MatAddCuda(float* A, float* B, float* out, int idx)
{
    out[idx] = A[idx]+B[idx];
}


__global__
void MatCalculate(float *A, float *B, float*out, int iteration_cnt)
{
    int idx = threadIdx.x;
    int i=0, j=0;
    //for (i = 0; i < ITERATION_NUM*2; i++) {
    for (i = 0; i < iteration_cnt; i++) {
        for (j= 0; j < iteration_cnt; j++)
            MatAddCuda(A, B, out, idx);
    }
}


void cudaTestWrapper_MatCalculate(float* A, float* B, int size, int calType, float* out, int iteration_cnt)
{
    CHECKCUDA(cudaMemcpy(dev_A, A, size* sizeof(float), cudaMemcpyHostToDevice));
    CHECKCUDA(cudaMemcpy(dev_B, B, size* sizeof(float), cudaMemcpyHostToDevice));

    dim3 numBlocks(1);
    dim3 threadsPerBlock(size);

    switch(calType)
    {
        case 1: // sum
            //cout << "cuda sum " << size << endl;
            MatCalculate<<<numBlocks, threadsPerBlock, 0>>>(dev_A,dev_B,dev_out, iteration_cnt);
            break;
        case 2: // multiplication
            break;
        default :
            break;
    }
    cudaThreadSynchronize();

    CHECKCUDA(cudaMemcpy(out, dev_out, size * sizeof(float), cudaMemcpyDeviceToHost));
}


void cudaInit(int size)
{
    // Copying structure hue and sat to cudaMemCpy
    CHECKCUDA(cudaMalloc(&dev_A, size * sizeof(float)));
    CHECKCUDA(cudaMalloc(&dev_B, size * sizeof(float)));
    CHECKCUDA(cudaMalloc(&dev_out, size * sizeof(float)));

    CHECKCUDA(cudaMallocHost((float **) &matrix1, size* sizeof(float)));
    CHECKCUDA(cudaMallocHost((float **) &matrix2, size* sizeof(float)));
    CHECKCUDA(cudaMallocHost((float **) &outBuffer, size* sizeof(float)));
}

void cudaExit()
{
    //cout << "cudaExit" << endl;
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_out);

    cudaFreeHost(matrix1);
    cudaFreeHost(matrix2);
    cudaFreeHost(outBuffer);
}
