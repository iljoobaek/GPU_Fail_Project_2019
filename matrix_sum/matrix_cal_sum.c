#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/time.h>
#include <sys/syscall.h>
#include <sched.h>
#include <pthread.h>
#include <string.h>

using namespace std;


#define MAX_MATRIX_SIZE 200
#define MILLISEC_TO_MICROSEC 1000
#define MILLISEC_TO_NANOSEC 1000000

extern float* matrix1;
extern float* matrix2;
extern float* outBuffer;
extern void cudaTestWrapper_MatCalculate(float* A, float* B, int size, int calType, float* out, int iteration_cnt);
extern void cudaInit(int size);
extern void cudaExit();


#define MATRIX_CAL_SUM_DEBUG
#ifdef MATRIX_CAL_SUM_DEBUG
	#define MatCalSum_DPNT(fmt, args...)		fprintf(stdout, fmt, ## args)
	#define MatCalSum_EPNT(fmt, args...)		fprintf(stderr, fmt, ## args)
#else
	#define MatCalSum_DPNT(fmt, args...)
	#define MatCalSum_EPNT(fmt, args...)		fprintf(stderr, fmt, ## args)
#endif

// workload parameters
long int C_CPU = 50, C_GPU = 500;

unsigned long long current_time()
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return t.tv_sec * 1000000000ULL + t.tv_nsec;
}


int initialize(int size)
{

    cudaInit(size);

    return 0;
}


void dummyCalulationCPU(unsigned long long C) // ms
{
    long int i = 0, dummyValue = 0;
    unsigned long long start_time, end_time;
    unsigned long long computeTime = 0;
    C = C*MILLISEC_TO_NANOSEC;


    while(1) {
        start_time = current_time();
        for (i = 0; i<10000; i++) {
            dummyValue = dummyValue + 111;
            dummyValue = dummyValue * 111;
        }
        dummyValue = 0;
        end_time = current_time();
        computeTime += end_time-start_time;
        if (computeTime >= C/2) {
            MatCalSum_DPNT("[Mat_Cal_Sum] cpu compute Time : %llu ms\n", computeTime/MILLISEC_TO_NANOSEC);
            break;
        }
    }
}

void periodicTask ()
{
    while(1)
    {
        // CPU 1 execution
        MatCalSum_DPNT ("[Mat_Cal_Sum] CPU execution 1\n");
        dummyCalulationCPU(C_CPU/2);

        #ifdef TIME_COST_ANALYSIS
        timeTrace_done(SECTION_CPU1);
        #endif /* TIME_COST_ANALYSIS*/

        // GPU execution
        MatCalSum_DPNT ("[Mat_Cal_Sum] GPU execution\n");
        // Initialize output buffer
        memset(outBuffer, 0, MAX_MATRIX_SIZE);
        cudaTestWrapper_MatCalculate(matrix1, matrix2, MAX_MATRIX_SIZE, 1, outBuffer, C_GPU);

        // CPU 2 execution
        MatCalSum_DPNT ("[Mat_Cal_Sum] CPU execution 2\n\n");
        dummyCalulationCPU(C_CPU/2);
    }
}


int main(int argc, char *argv[])
{
    int i = 0;

    MatCalSum_DPNT("[Mat_Cal_Sum] initialize start\n");
    if (initialize(MAX_MATRIX_SIZE) < 0) {
        return -1;
    }

    // initialize matrix
    for(i=0; i<MAX_MATRIX_SIZE; i++ )
    {
        matrix1[i] = i;
        matrix2[i] = i;
        outBuffer[i] = 0;
    }
    MatCalSum_DPNT("[Mat_Cal_Sum] CPU->GPU initialize done\n\n");


    periodicTask();

    cudaExit();


}
