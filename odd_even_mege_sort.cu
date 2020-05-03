#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <cmath>

using namespace std;
#define THREAD 512
#define BLOCK 4096
#define SWAP(x,y) { x = x + y; y = x - y; x = x - y; }

//odd
__global__ void odd_merge(int* data, int index) {
    int i;
    //distinct task id 
    i = threadIdx.x + blockDim.x * blockIdx.x;
   
    //odd state
    if(index % 2 == 1 && index < BLOCK * THREAD - 1){
        if (data[i] > data[i + 1])
            SWAP(data[i], data[i + 1]);
    }
}
//even
__global__ void even_merge(int* data, int index) {
    int i;
    //distinct task id 
    i = threadIdx.x + blockDim.x * blockIdx.x;

    //even state
    if (index % 2 == 0 && index < BLOCK * THREAD - 1) {
        if (data[i] > data[i + 1])
            SWAP(data[i], data[i + 1]);
    }
}

//set random data
void input_data(int* data, int size) {
    for (int i = 0; i < size; i++)
        data[i] = rand() % 100000;
}

int main() {
    //initialize
    int* dev_data;
    int size = BLOCK * THREAD;
    clock_t start;

    //allocate host memory
    int* data = (int*)malloc(size*sizeof(int));
    size_t dev_size = size * sizeof(int);

    //allocate device memory
    cudaMalloc((void**)&dev_data, dev_size);

    //input data : Number of THREAD*BLOCK
    input_data(data, size);

    //finished input data and memcpy host to device
    cudaMemcpy(dev_data, data, dev_size, cudaMemcpyHostToDevice);

    //start odd_even_merge_sort
    start = clock();
    int half = size / 2;

    dim3 block_dim(BLOCK, 1);
    dim3 thread_dim(THREAD, 1);

    for (int i = 0; i < half; i++) {
        even_merge << <block_dim, thread_dim, dev_size >> > (dev_data, size);
        if (i != half - 1)
            odd_merge << <block_dim, thread_dim, dev_size >> > (dev_data, size);
        if (i == half - 1 && size % 2 == 0)
            odd_merge << <block_dim, thread_dim, dev_size >> > (dev_data, size);
    }
    cout << "time for "<<BLOCK*THREAD<<" datas to sort: " << (double)(clock() - start) << "ms\n";

    //ended sorting and memcpy device to host
    cudaMemcpy(data, dev_data, dev_size, cudaMemcpyDeviceToHost);

    //free all memory
    cudaFree(dev_data);
    free(data);
    cout << "Odd_Even_Merge_Sort for CUDA";
    return 0;
}
