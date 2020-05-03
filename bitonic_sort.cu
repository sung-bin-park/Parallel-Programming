#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <cmath>

using namespace std;
#define THREAD 256
#define BLOCK 512
#define SWAP(x,y) { x = x + y; y = x - y; x = x - y; }

//call host
__global__ void bitonic_kernel(int* dev_size, int block, int thread) {
    int i, j;
    //distinct task id 
    i = threadIdx.x + blockDim.x * blockIdx.x;
    j = i ^ block;

    if (j > i) {
        if ((i & thread) == 0) {
            //sorting ascending
            if (dev_size[i] > dev_size[j])
                SWAP(dev_size[i], dev_size[j]);
        }
        if ((i & thread) != 0) {
            //sorting_descending
            if (dev_size[i] < dev_size[j])
                SWAP(dev_size[i], dev_size[j]);
        }
    }
}

//set random data
void input_data(int* data, int size) {
    for (int i = 0; i < size; i++)
        data[i] = rand() % 100000;
}
//do bitonic

void bitonic_sort(int* data, int block, int thread) {
    //initialize device size value
    int* dev_data;

    //initial array size
    int size = block * thread;

    //define device size
    size_t dev_size = size * sizeof(int);

    //memory allocateing to device
    cudaMalloc((void**)&dev_data, dev_size);
    cudaMemcpy(dev_data, data, dev_size, cudaMemcpyHostToDevice);

    //define block, thread for dim3 struct
    dim3 block_dim(block, 1);
    dim3 thread_dim(thread, 1);

    //bitonic sort
    for (int i = 2; i <= size; i <<= 1)
        for (int j = i >> 1; j > 0; j = j >> 1) {
            bitonic_kernel << <block_dim, thread_dim, dev_size >> > (dev_data, j, i);
        }
    //memory return
    cudaMemcpy(data, dev_data, dev_size, cudaMemcpyDeviceToHost);
    cudaFree(dev_data);
}

int main() {
    double sum = 0;
    //initialize host variables
    int* data3;
    //initialize clock_t variable
    clock_t start, end;
 
        data3 = (int*)malloc(BLOCK*THREAD * sizeof(int));
        input_data(data3, BLOCK * THREAD);

        start = clock();
        bitonic_sort(data3, BLOCK, THREAD);

        cout << "time for 131072 datas to sort: " << (double)(clock() - start) << "ms\n";
        free(data3);
    cout << "Bitonic_Sort for CUDA ";
    return 0;
}
