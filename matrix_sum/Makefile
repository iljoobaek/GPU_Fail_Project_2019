CXX=g++
NVCC=nvcc
CFLAGS=-O3 -Wno-unused-result -Wno-write-strings

CUDA_LIB_PATH=/usr/local/cuda/lib64

matrix_calculateCUDA: matrix_cal_sum.c cuda_matrix_cal_sum.o
	$(CXX) $(CFLAGS) matrix_cal_sum.c -o matrix_cal_sum cuda_matrix_cal_sum.o -L$(CUDA_LIB_PATH) -lcuda -lcudart

cuda_matrix_cal_sum.o: cuda_matrix_cal_sum.cu
	$(NVCC) --lib cuda_matrix_cal_sum.cu -o cuda_matrix_cal_sum.o


clean:
	rm -f *~ matrix_cal_sum *.o
