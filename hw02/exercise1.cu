#include <stdio.h>
#define NB 1
#define TPB 256

__global__ void hello()
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
//  printf("blockIdx is %2d, blockDim is %2d, threadIdx is %2d \n", blockIdx.x, blockDim.x, threadIdx.x);
  printf("Hello World! My thread ID is %2d \n", i);
}

int main()
{

  // Launch kernel to print "Hello World"
  hello<<<NB, TPB>>>();
  cudaDeviceSynchronize();
  
  return 0;
}
