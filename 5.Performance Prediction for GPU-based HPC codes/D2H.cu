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

	for (int i = 0; i<size/4;i++){
		host[i]=rand();
	}


	cudaMemcpy(device,host,sizeof(float)*size,cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();



    gettimeofday(&start_timeS,NULL);
    for (int i = 0; i<30; i++){
    cudaMemcpy(host,device,sizeof(float)*size,cudaMemcpyDeviceToHost);
    }
    gettimeofday(&end_timeE,NULL);

    cudaDeviceSynchronize();


	float timeM = ((end_timeE.tv_sec*1000000+end_timeE.tv_usec)-(start_timeS.tv_sec*1000000+start_timeS.tv_usec));
		
	printf("size: %d k, time from device to host : %f ms\n",a,timeM/30);

}
