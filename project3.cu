#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <chrono>
#include <time.h>
#include <iostream>
#include <cfloat>
#include <algorithm>  // not sure if needed

//for CUDA garbage - maybe useful
#include <cuda_runtime.h>


//#define int LENGTH 100     // length of data set - okay this doesnt work aparently

int LENGTH = 10000;


//this is possible error, not save mem correctly
#if __CUDA_ARCH__ < 600  // allows us to use atomicMin with doubles on our old as dirt CUDA versions
__device__ double atomicMinf(double* address, double val){

    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    while (val < __longlong_as_double(old)) {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,  __double_as_longlong(val)); // (old == assumed ? val : old)
    }
    return __longlong_as_double(old);
}
#endif

//CUDA kernal for part one
__global__ void findLeast(const double *array, double *m, const int size){
  extern __shared__ double share[]; // extern : "size determined at runtime by the kernel's caller via a launch configuration argument" - whatever that means
  int tid = threadIdx.x;
  int gid = blockDim.x * blockIdx.x + threadIdx.x; // what element to work on
  share[tid] = DBL_MAX; // initalize with largest num

  while(gid < size){  //check to see if in range
    share[tid] = max(share[tid], array[gid]);
    gid += gridDim.x*blockDim.x;  // what element each thread should work on
  }
  __syncthreads();
  gid = blockDim.x * blockIdx.x + threadIdx.x; // reset gid for future calculations

  // reduce spread out shared memory in block into one location: shared[0]
  for(int i = blockDim.x / 2; i > 0; i/=2){   // note blockDim.x is number of threads in a block, always even
    if(tid < i and gid < size)
      share[tid] = max(share[tid], share[tid + i]);
    __syncthreads();
  }

  // compare across blocks to find true min
  if(tid == 0)
    atomicMinf(m, share[0]);    // or is the error here, no clue
}

///////////////////////////////////////////////////////////////////////////////

void fillArray(double *n, int s){
  std::srand(std::time(NULL));    // lets seed rand with time for fun!!
  for(int i = 0; i < s; i++){
    n[i] =  (double)std::rand();
  }
}

void printArray(float *n, int s){
  for(int i = 0; i < s; i++){
    printf("%.5f ", n[i]);
  }
}

//finish
double checkMin(double* a, int size){
  double min = DBL_MAX;

  return min;
}

int main(int argc, char **argv){

  size_t size = LENGTH * sizeof(double);
  double *h_a = (double*)malloc(size); // allocate mem for host array
  double *output = (double*)malloc(sizeof(double)); // allocate memory for output
  cudaError_t err = cudaSuccess; // error check, maybe gets implemented

  if(h_a == NULL || output == NULL){
    fprintf(stderr, "Failed to allocate main memory");
    exit(EXIT_FAILURE);
  }
  // fill the array with random values, fill output with max value
  fillArray(h_a, LENGTH);
  *output = DBL_MAX;

  // allocate memory on device for input vector a
  double *d_a = NULL;
  err = cudaMalloc(&d_a, size);
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector a (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  // allocate memory on device for output o
  double *d_o = NULL;
  err = cudaMalloc(&d_o, sizeof(double));
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate output (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // copy main memory data into device memory
  err = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to copy vector a from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(d_o, output, sizeof(double), cudaMemcpyHostToDevice);
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to copy output from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // ready for takeoff, luanch CUDA kernal
  int threadsPerBlock = 1024;
  int blocksPerGrid = (LENGTH + threadsPerBlock - 1) / threadsPerBlock;
  findLeast<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_o, LENGTH);
  err = cudaGetLastError();
  if(err != cudaSuccess){
    fprintf(stderr,"(error code %s)\nYou done screwed up\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(&output, d_o, sizeof(double), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to copy device d_o to host output (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  //this should be where we verify the output

  printf("The minimum number: %f\n", *output);

  cudaFree(d_a);
  cudaFree(d_o);
  free(h_a);
  free(output);
  return 0;
}
