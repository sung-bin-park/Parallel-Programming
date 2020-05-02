#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include<stdlib.h>
#include<iostream>
using namespace std;

#define SIZE 1024
//call from host func (like order to gpu)
__global__ void vectoradd(int* a, int* b, int* c,int n) {
	int i = threadIdx.x;//to distinct thread
	
	for (i = 0; i < n; i++)c[i] = a[i] + b[i];
	//do sum of each thread
}

int main() {
	int* a, * b, * c;//host variable
	int* d_a, * d_b, * d_c;//device variable

	//malloc to host memory
	a = (int*)malloc(SIZE * sizeof(int));
	b = (int*)malloc(SIZE * sizeof(int));
	c = (int*)malloc(SIZE * sizeof(int));
	
	//malloc to device memory
	cudaMalloc(&d_a, SIZE * sizeof(int));
	cudaMalloc(&d_b, SIZE * sizeof(int));
	cudaMalloc(&d_c, SIZE * sizeof(int));

	//initialize variables
	for (int i = 0; i < SIZE; i++) {
		a[i] = rand() % 1000;
		b[i] = rand() % 1000;
	}

	//memcopy to device variables
	cudaMemcpy(d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	
	//call vectoradd func using 1 block, 1024 threads
	vectoradd <<< 1, SIZE >>> (d_a, d_b, d_c, SIZE);

	//memcopy to host, save device to host 
	cudaMemcpy(a, d_a, SIZE * sizeof(int), cudaMemcpyDeviceToHost); 
	cudaMemcpy(b, d_b, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(c, d_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	//print result of c variable
	for (int i = 0; i < SIZE; i++)
		cout << "c[" << i << "]=" << c[i] << "\n";
	//memory free to host
	free(a);
	free(b);
	free(c);
	//memory free to device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;


}