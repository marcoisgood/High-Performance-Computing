#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<cuda.h>
#include<time.h>
#include<sys/time.h>
#include<math.h>
int main(int argc, char ** argv){


	float *host, *device; 
	struct timeval start_timeS,end_timeE;
	int size;
	
	int a = atof(argv[1]);	
	size = a*1024; 
	host = (float*)malloc(sizeof(float)*size);
	cudaMalloc((void**)&device,sizeof(float)*size);
	/*host = (float*)malloc(sizeof(float)*2048);
	cudaMalloc((void**)&device,sizeof(float)*2048);
	host = (float*)malloc(sizeof(float)*4096);
	cudaMalloc((void**)&device,sizeof(float)*4096);
	host = (float*)malloc(sizeof(float)*8192);
	cudaMalloc((void**)&device,sizeof(float)*8192);
	host = (float*)malloc(sizeof(float)*16384);
	cudaMalloc((void**)&device,sizeof(float)*16384);
	host = (float*)malloc(sizeof(float)*32768);
	cudaMalloc((void**)&device,sizeof(float)*32768);
	host = (float*)malloc(sizeof(float)*65536);
	cudaMalloc((void**)&device,sizeof(float)*65536);
	host = (float*)malloc(sizeof(float)*131072);
	cudaMalloc((void**)&device,sizeof(float)*131072);
	host = (float*)malloc(sizeof(float)*262144);
	cudaMalloc((void**)&device,sizeof(float)*262144);
	host = (float*)malloc(sizeof(float)*524288);
	cudaMalloc((void**)&device,sizeof(float)*524288);
	*/
	for (int i = 0; i<size/4;i++){
		host[i]=rand();
	}

	
	gettimeofday(&start_timeS,NULL);
	for (int i = 0; i<30; i++){
	cudaMemcpy(device,host,sizeof(float)*size,cudaMemcpyHostToDevice);
        }
	gettimeofday(&end_timeE,NULL);
	
	cudaDeviceSynchronize();

	float timeM = ((end_timeE.tv_sec*1000000+end_timeE.tv_usec)-(start_timeS.tv_sec*1000000+start_timeS.tv_usec));
		
	printf("size: %d k, time Host to Device: %f ms\n",a,timeM/30);

}
