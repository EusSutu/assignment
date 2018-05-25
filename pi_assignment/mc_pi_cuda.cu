/////////////////////////////////////////////////////////////////////////
//                                                                     //
//  CUDA code which calculates PI using Monte-Carlo method             //
//  It will get random points in the square between (0,0) and (1,1)    //
//  Find whether it is in the circle of radius 1 and use it to find PI //
//  Name: Eusebiu Sutu                                                 //
//  E-mail: eusebiu.sutu@lincoln.ox.ac.uk                              //
//  Date: May 25th, 2018                                               //
//  CWM: High performance computing                                    //
//                                                                     //
/////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>

#define PI 3.1415926536

//Kernel which verifies whether points are in the circle
//If it is then add to area

__global__ void mc_pi (int *area_d,float *f_rand_d,int N)
{
  float r2; //Distance squared from point to origin
  int index = 2*(blockIdx.x*N+threadIdx.x);
  float x,y;

  //Coordinates are stored as {point0.x,point0.y,point1.x,point1.y,...}
  x = f_rand_d[index];
  y = f_rand_d[index+1];

  r2 = x*x + y*y;
  if(r2 <= 1)
    atomicAdd(area_d, 1); 
}

//Function which inputs different values to the number of points and calculates pi and th error

void error(curandGenerator_t gen)
{
   float *f_rand_d,pi;  //f_rand_d are the coordinates generated, pi will be the value for PI calculated  
   int i;               //i*i is the total number of points generated
   int area,*area_d;    //area_d is the area calculated on the device, area is a host copy of it

   for (i=2;i<=1024;i*=2)
   {
     area = 0;          // reset
     cudaMalloc( (void **) &f_rand_d, 2*i*i*sizeof(float));           // allocate device memory
     cudaMalloc( (void **) &area_d, sizeof(int));

     cudaMemcpy(area_d, &area, sizeof(int), cudaMemcpyHostToDevice);  // reset area_d

     curandGenerateUniform( gen, f_rand_d, 2*i*i);                    //generate the coordinates

     mc_pi<<<i,i>>>(area_d,f_rand_d,i);                               //calculate area

     cudaMemcpy(&area, area_d, sizeof(int), cudaMemcpyDeviceToHost);  //get result back to host

     pi = (float)area/((float)i*(float)i);                            //calculate pi from area
     pi*=4;

     printf("i=%d  pi=%f error=%f\n",i,pi,pi-PI);                     //print results out on the console

     cudaFree(f_rand_d);                                              //free memory
     cudaFree(area_d);
   }
}

int main()
{
  //Initialize the GPU
  
  int deviceid=0;
  int devCount;
  cudaGetDeviceCount(&devCount);
  if(deviceid<devCount) cudaSetDevice(deviceid);
  else return 1;
  

  //Create the generator
  curandGenerator_t gen;  
  curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed( gen, 1234ULL);

  //Use the error funtion
  error(gen);

  cudaDeviceReset();
}
