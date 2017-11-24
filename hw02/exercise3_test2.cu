#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "helper_math.h"
#include "curand_kernel.h"

#define NUM_PARTICLES 10000
#define NUM_ITERATIONS 100000
#define BLOCK_SIZE 64


typedef struct Particle {
	float3 pos;
	float3 vel;
} Particle;
	

__device__ float3 update(float3 DERIV, Particle p)
{
  return DERIV * p.vel + p.pos;
}

__global__ void Particles_Kernel(unsigned seed, float3 DERIV, Particle *gpu_particles, float3 *gpu_rand_vel)
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  gpu_particles[i].vel = gpu_rand_vel[i];
  gpu_particles[i].pos = update(DERIV, gpu_particles[i]);
}

int main()
{
  int err_flag=0;
  time_t t;
  unsigned int eclapsed_time_cpu = 0, eclapsed_time_gpu = 0;
  const float3 DERIV = make_float3(0.3, 0.4, 0.5);
  
  Particle *cpu_particles = (Particle*)malloc(NUM_PARTICLES * sizeof(Particle));
  float3 *cpu_rand_vel = (float3*)malloc(NUM_PARTICLES * sizeof(float3));
  Particle *gpu_particles_copy = (Particle*)malloc(NUM_PARTICLES * sizeof(Particle));
  
  memset(cpu_particles, 0, NUM_PARTICLES * sizeof(Particle));

  time(&t);
  srand((unsigned)t);

  struct timeval tv_start;
  struct timeval tv_stop;

    // Declare a pointer for an array of floats
  Particle *gpu_particles = 0;
  float3 *gpu_rand_vel = 0;

  // Allocate device memory to store the output array
  cudaMalloc(&gpu_particles, NUM_PARTICLES * sizeof(Particle));
  cudaMalloc(&gpu_rand_vel, NUM_PARTICLES * sizeof(float3));
  cudaMemset(&gpu_particles, 0, NUM_PARTICLES * sizeof(Particle));
  
  for (int j = 0; j < NUM_ITERATIONS; j++)
  { 
        gettimeofday(&tv_start,NULL);
	for (int i = 0; i < NUM_PARTICLES; i++)
	{
		cpu_particles[i].vel.x = rand()/(float)(RAND_MAX);
		cpu_particles[i].vel.y = rand()/(float)(RAND_MAX);
		cpu_particles[i].vel.z = rand()/(float)(RAND_MAX);
	}
	for (int i = 0; i < NUM_PARTICLES; i++)
		cpu_particles[i].pos = cpu_particles[i].vel * DERIV + cpu_particles[i].pos;

  	gettimeofday(&tv_stop,NULL);
  	eclapsed_time_cpu += (tv_stop.tv_sec - tv_start.tv_sec)*1000000 + (tv_stop.tv_usec - tv_start.tv_usec);
  
	if (j == NUM_ITERATIONS - 1)
  		printf("Computing SAXPY on the CPU… Done! Time used is %d us.\n", eclapsed_time_cpu);			
  
	for (int i = 0; i < NUM_PARTICLES; i++)
		cpu_rand_vel[i] = cpu_particles[i].vel;
 
	cudaMemcpy(gpu_rand_vel, cpu_rand_vel, NUM_PARTICLES * sizeof(float3), cudaMemcpyHostToDevice);

  	// Launch kernel to compute and store distance values
  	Particles_Kernel <<<(NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >>>(t, DERIV, gpu_particles, gpu_rand_vel);	
        
  	cudaMemcpy(gpu_particles_copy, gpu_particles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);  
	
  	gettimeofday(&tv_stop,NULL);
  	eclapsed_time_gpu += (tv_stop.tv_sec - tv_start.tv_sec)*1000000 + (tv_stop.tv_usec - tv_start.tv_usec);

  } 
 
  gettimeofday(&tv_start,NULL);

  cudaDeviceSynchronize();
  cudaFree(gpu_particles); // Free the memory

  gettimeofday(&tv_stop,NULL);
  eclapsed_time_gpu += (tv_stop.tv_sec - tv_start.tv_sec)*1000000 + (tv_stop.tv_usec - tv_start.tv_usec);
  
  printf("Computing SAXPY on the GPU… Done! Time used is %d us.\n", eclapsed_time_gpu);
  

  for (int i = 0; i<NUM_PARTICLES; i++)
  {
	if (abs(gpu_particles_copy[i].pos.x - cpu_particles[i].pos.x) > 0.001 * NUM_ITERATIONS || abs(gpu_particles_copy[i].pos.y - cpu_particles[i].pos.y) > 0.001 * NUM_ITERATIONS || abs(gpu_particles_copy[i].pos.z - cpu_particles[i].pos.z) > 0.001 * NUM_ITERATIONS)
	{
		printf("Comparing the output for each implementation... incorrect!\n");
		printf("CPU is: %f, %f, %f;\n", cpu_particles[i].pos.x, cpu_particles[i].pos.y, cpu_particles[i].pos.z);
		printf("GPU is: %f, %f, %f;\n", gpu_particles_copy[i].pos.x, gpu_particles_copy[i].pos.y, gpu_particles_copy[i].pos.z);
		err_flag=1;
		break;
	}
  }
 
  if (!err_flag)
  	printf("Comparing the output for each implementation... Correct!\n");

  return 0;
}
