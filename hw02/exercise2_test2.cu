#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define ARRAY_SIZE 10000
#define TPB 256

__device__ float SAXPY(float A, float x, float y)
{
  return A*x+y;
}

__global__ void SAXPY_Kernel(float A, int iter_num, float *gpu_X, float *gpu_Y, float *gpu_results)
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  for(int j=0; j<iter_num; j++)
  	gpu_results[i] = SAXPY(A, gpu_X[i], gpu_Y[i]);
}

int main(int argc, char *argv[])
{
  int iter_num = atoi(argv[1]);
  int err_flag = 0;
  unsigned int eclapsed_time;
  const float A = 0.5f;
  
  float *cpu_X= (float*)malloc(ARRAY_SIZE*sizeof(float));
  float *cpu_Y= (float*)malloc(ARRAY_SIZE*sizeof(float));
  float *results= (float*)malloc(ARRAY_SIZE*sizeof(float));
  
  srand((unsigned)time(0));

  for(int i=0; i<ARRAY_SIZE; i++)
  {
	cpu_X[i]= rand()/(float)(RAND_MAX);
	cpu_Y[i]= rand()/(float)(RAND_MAX);
  }

  struct timeval tv_start;
  struct timeval tv_stop;
  gettimeofday(&tv_start,NULL);

  for(int j=0; j<iter_num; j++)
  	for(int i=0; i<ARRAY_SIZE; i++)
  		results[i]=A*cpu_X[i]+cpu_Y[i]; 

  gettimeofday(&tv_stop,NULL);
  eclapsed_time = (tv_stop.tv_sec - tv_start.tv_sec)*1000000 + (tv_stop.tv_usec - tv_start.tv_usec);
  
  printf("Computing SAXPY on the CPU… Done! Time used is %d us.\n", eclapsed_time);		
  
  // Declare a pointer for an array of floats
  float *gpu_X = 0;
  float *gpu_Y = 0;
  float *gpu_results = 0;
  
  // Allocate device memory to store the output array
  cudaMalloc(&gpu_X, ARRAY_SIZE*sizeof(float));
  cudaMalloc(&gpu_Y, ARRAY_SIZE*sizeof(float));
  cudaMalloc(&gpu_results, ARRAY_SIZE*sizeof(float));
  
  gettimeofday(&tv_start,NULL);

  // code for test 2, host to device memcopy only once, looping in kernal.
  
  cudaMemcpy(gpu_X, cpu_X, ARRAY_SIZE*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_Y, cpu_Y, ARRAY_SIZE*sizeof(float),cudaMemcpyHostToDevice);
  
  // Launch kernel to compute and store distance values
  SAXPY_Kernel<<<(ARRAY_SIZE+TPB-1)/TPB, TPB>>>(A, iter_num, gpu_X, gpu_Y, gpu_results);

  cudaDeviceSynchronize();
  

  gettimeofday(&tv_stop,NULL);

  eclapsed_time = (tv_stop.tv_sec - tv_start.tv_sec)*1000000 + (tv_stop.tv_usec - tv_start.tv_usec);

  cudaMemcpy(cpu_Y, gpu_results, ARRAY_SIZE*sizeof(float),cudaMemcpyDeviceToHost);
  
  cudaFree(gpu_X); // Free the memory
  cudaFree(gpu_Y); // Free the memory
  cudaFree(gpu_results); // Free the memory
  
  printf("Computing SAXPY on the GPU… Done! Time used is %d us.\n", eclapsed_time);

  for(int i=0; i<ARRAY_SIZE; i++)
  {
	if (results[i] != cpu_Y[i])
	{
		printf("Comparing the output for each implementation… incorrect!\n");
		err_flag=1;
		break;
	}
  }
 
  if (!err_flag)
  	printf("Comparing the output for each implementation… Correct!\n");

  return 0;
}
