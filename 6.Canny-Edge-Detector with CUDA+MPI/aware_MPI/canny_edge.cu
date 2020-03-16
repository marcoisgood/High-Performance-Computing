#include<stdio.h>
#include <string.h>
#include "stdlib.h"
#include<math.h>
#include"time.h"
#include"sys/time.h"
#include<cuda.h>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include<thrust/sort.h>
#include<thrust/copy.h>
#include<mpi.h>
#include"canny_edge.h"
/*


struct location {
int locationI;
int locationJ;

};

int compare(const void *a, const void *b){

float c = *(float*)a;
float d = *(float*)b;
if(c < d) return -1;
else return 1;
}


void print_matrix(float *matrix, int height, int width){
for(int i=0; i<height; i++){
for(int j=0; j<width; j++){
printf("%.3f ", *(matrix+(i*width)+j));
}
printf("\n");
}
}

int range(int y, int x, int h, int w){
if(x < 0 || x >= w){
return 0;
}
else if(y < 0 || y >= h){
return 0;
}
else
return 1;
}
*/


float *ghost_chunk(float *image, int height, int width, int a, int comm_size, int comm_rank){
    MPI_Status status;
    float *output;
    if(comm_rank == 0 || comm_rank == (comm_size-1))
       // output = (float*)malloc(sizeof(float)*width*(height + a));
	
	cudaMalloc((void **)&output,sizeof(float)*width*(height + a));
	
    else
	
	cudaMalloc((void **)&output,sizeof(float)*width*(height + 2*a));
	
       // output = (float*)malloc(sizeof(float)*width*(height + 2*a));

    if(comm_rank == 0){

        MPI_Sendrecv(image+width*(height-a), a*width, MPI_FLOAT, comm_rank+1, comm_rank,
                     output+width*height, a*width, MPI_FLOAT, comm_rank+1, comm_rank+1, MPI_COMM_WORLD, &status);

        memcpy(output, image, sizeof(float)*width*height);
        return output;

    }else if(comm_rank == (comm_size-1)){

        MPI_Sendrecv(image, a*width, MPI_FLOAT, comm_rank-1, comm_rank,
                     output, a*width, MPI_FLOAT, comm_rank-1, comm_rank-1, MPI_COMM_WORLD, &status);
        memcpy(output+a*width, image, sizeof(float)*width*height);

        return output+a*width;

    }else{

        //send top data to previous rank, receive top data from previous rank
        MPI_Sendrecv(image, a*width, MPI_FLOAT, comm_rank-1, comm_rank,
                     output, a*width, MPI_FLOAT, comm_rank-1, comm_rank-1, MPI_COMM_WORLD, &status);

        //send bottom data to next rank, receive top data from next rank
        MPI_Sendrecv(image+width*(height-a), a*width, MPI_FLOAT, comm_rank+1,comm_rank,
                    output+width*(height+a), a*width, MPI_FLOAT, comm_rank+1, comm_rank+1, MPI_COMM_WORLD, &status);

        memcpy(output+a*width, image, sizeof(float)*width*height);
        return output+a*width;
    }


}



__global__
void convolveGPU(float *image, float *outputG, float *kernel, int height,int width, int k_height, int k_width, int top, int bottom){

//using global memory
int i,j,m,offseti,offsetj;
float kerw=(k_width>k_height)?k_width:k_height;
//printf("%f",kerw);
i=threadIdx.x+blockIdx.x*blockDim.x;
j=threadIdx.y+blockIdx.y*blockDim.y;

if(i<height && j<width ){
float sum = 0;
for( m=0; m<kerw; m++){
offseti = k_height>1?(-1*(k_height/2)+m):0;
offsetj = k_width>1?(-1*(k_width/2)+m):0;
if( (i+offseti)>=0-top && (i+offseti)<height+bottom && (j+offsetj)>=0 && (j+offsetj)< width)
sum+= image[(i+offseti)*width+(j+offsetj)]*kernel[m];
}
outputG[i*width+j]=sum;
}
/*
//using shared memory
int m,offseti,offsetj;
float kerw=(k_width>k_height)?k_width:k_height;
//printf("%f",kerw);
int locaIx = threadIdx.x;
int locaIy = threadIdx.y;
int globIx = blockIdx.x*blockDim.x+ threadIdx.x;
int globIy = blockIdx.y*blockDim.y+ threadIdx.y;

//read global memory to shared memory
extern __shared__ float AShared[];
AShared[locaIx*blockDim.y+locaIy]=image[globIx*width+globIy];
__syncthreads();

if(globIx<height && globIy<width ){
float sum = 0;

for( m=0; m<kerw; m++){
offseti = k_height>1?(-1*(k_height/2)+m):0;
offsetj = k_width>1?(-1*(k_width/2)+m):0;

if( (locaIx+offseti)>=0 && (locaIx+offseti)< blockDim.x && (locaIy+offsetj)>=0 && (locaIy+offsetj)<blockDim.y)
sum+= AShared[(locaIx+offseti)*blockDim.y+(locaIy+offsetj)]*kernel[m];

else if((globIx+offseti) >= 0 && (globIx+offseti) < height && (globIy+offsetj)>=0 && (globIy+offsetj)<width)


sum+= image[(globIx+offseti)*width+(globIy+offsetj)]*kernel[m];


}
output[globIx*width+globIy]=sum;
}
*/
}



void convolve(float *chunk_image,float *temp_horizon,float *gauss_Kernel,float *dgauss_Kernel,int height,int width, int k_height, int k_width ,int comm_size,int comm_rank,int blocksize){ 
/*
void convolve(float *image, float *d_temp_horizon, float *kernel, int height,
              int width, int k_height, int k_width, int comm_size, int comm_rank, int blocksize){
  */  





  	int w = k_height>1?k_height:k_width;
	int a = floor(w/2);

	float *d_temp_horizon,*d_temp_horizon_S,*d_chunk_image,*d_GKernel, *d_DKernel, *chunk_image_ghost, *chunk_image_ghost_S;
	
	if(comm_rank == 0 || comm_rank == (comm_size-1))
	cudaMalloc((void **)&d_chunk_image,sizeof(float)*width*(height+a));
	
	else
	cudaMalloc((void **)&d_chunk_image,sizeof(float)*width*(height+2*a));
	

	cudaMalloc((void **)&d_DKernel,sizeof(float)*w);
	cudaMalloc((void **)&d_temp_horizon,sizeof(float)*width*height);
	cudaMalloc((void **)&d_GKernel,sizeof(float)*w);
	cudaMalloc((void **)&d_temp_horizon_S,sizeof(float)*width*height);





	if(comm_rank == 0 || comm_rank == (comm_size-1))
	cudaMemcpy(d_chunk_image,chunk_image,sizeof(float)*width*(height+a),cudaMemcpyHostToDevice);
	else
	cudaMemcpy(d_chunk_image,chunk_image,sizeof(float)*width*(height+2*a),cudaMemcpyHostToDevice);

	cudaMemcpy(d_GKernel,gauss_Kernel,sizeof(float)*w,cudaMemcpyHostToDevice);


	cudaMemcpy(d_DKernel,dgauss_Kernel,sizeof(float)*w,cudaMemcpyHostToDevice);


	cudaDeviceSynchronize();

	dim3 dimBlock(blocksize,blocksize,1);
	dim3 dimGrid(ceil(height/blocksize),ceil(width/blocksize),1);

	int top, bottom;
    
   	 if (comm_rank == 0){
        	top = 0;
        	bottom = floor(k_height/2);
    	}else if (comm_rank == comm_size-1){
        	top = floor(k_height/2);
        	bottom = 0;
    	}else{
        	top = floor(k_height/2);
        	bottom = top;
    	}	
 	chunk_image_ghost = ghost_chunk(d_chunk_image, height, width, a ,comm_size, comm_rank);

//	convolveGPU<<<dimGrid,dimBlock,sizeof(float)*blocksize*blocksize>>>(chunk_image_ghost, d_temp_horizon, d_GKernel, height, width, k_height, k_width, top, bottom);   
       
//	chunk_image_ghost_S = ghost_chunk(d_temp_horizon, height, width, a ,comm_size, comm_rank);
	
//	convolveGPU<<<dimGrid,dimBlock,sizeof(float)*blocksize*blocksize>>>(chunk_image_ghost_S, d_temp_horizon_S, d_DKernel, height, width,  k_width,k_height, top, bottom);	

	cudaMemcpy(temp_horizon,d_temp_horizon_S,sizeof(float)*width*height,cudaMemcpyDeviceToHost);
	    }








__global__
void MagnitudeGPU(float *vertical, float *horizon, float *Mag, int height, int width){

int i,j;
i=threadIdx.x+blockIdx.x*blockDim.x;
j=threadIdx.y+blockIdx.y*blockDim.y;

if(i<height && j<width)
Mag[i*width+j]=sqrt(pow(vertical[i*width+j],2)+pow(horizon[i*width+j],2));

}


void Magnitude(float *vertical, float *horizon,float *d_Mag, int height, int width, int blocksize){
	

	float *d_vertical,*d_horizon;
	cudaMalloc((void **)&d_Mag,sizeof(float)*width*height);
 	cudaMalloc((void **)&d_vertical,sizeof(float)*width*height);  
 	cudaMalloc((void **)&d_horizon,sizeof(float)*width*height);
	
	cudaMemcpy(d_vertical,vertical,sizeof(float)*width*height,cudaMemcpyHostToDevice);
	cudaMemcpy(d_horizon,horizon,sizeof(float)*width*height,cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();


	dim3 dimBlock(blocksize,blocksize,1);
	dim3 dimGrid(ceil(height/blocksize),ceil(width/blocksize),1);


	MagnitudeGPU<<<dimGrid,dimBlock>>>(d_vertical, d_horizon, d_Mag, height, width);   
 	cudaDeviceSynchronize();

 
	//cudaMemcpy(Mag,d_Mag,sizeof(float)*width*height,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();


	cudaFree(d_Mag);
	cudaFree(d_vertical);
	cudaFree(d_horizon);

}



__global__
void DirectionGPU(float *vertical, float *horizon, float *Dir, int height,int width){

int i,j;
i=threadIdx.x+blockIdx.x*blockDim.x;
j=threadIdx.y+blockIdx.y*blockDim.y;
if(i<height&&j<width)
Dir[i*width+j]=atan2(vertical[i*width+j],horizon[i*width+j]);


}


void Direction(float *vertical, float *horizon, float *d_Dir, int height,int width, int blocksize){
    
 
	float *d_vertical,*d_horizon;
	cudaMalloc((void **)&d_Dir,sizeof(float)*width*height);
 	cudaMalloc((void **)&d_vertical,sizeof(float)*width*height);  
 	cudaMalloc((void **)&d_horizon,sizeof(float)*width*height);
	
	cudaMemcpy(d_vertical,vertical,sizeof(float)*width*height,cudaMemcpyHostToDevice);
	cudaMemcpy(d_horizon,horizon,sizeof(float)*width*height,cudaMemcpyHostToDevice);
	
	dim3 dimBlock(blocksize,blocksize,1);
	dim3 dimGrid(ceil(height/blocksize),ceil(width/blocksize),1);


	DirectionGPU<<<dimGrid,dimBlock>>>(d_vertical, d_horizon, d_Dir, height, width);   
  
	//cudaMemcpy(Dir,d_Dir,sizeof(float)*width*height,cudaMemcpyDeviceToHost);
   
       
	cudaFree(d_Dir);
	cudaFree(d_vertical);
	cudaFree(d_horizon);

    
       
}



__global__
void supressionGPU(float *sup, float *Mag, float *Dir, int width, int height, int top, int bottom){

int i,j;
i=threadIdx.x+blockIdx.x*blockDim.x;
j=threadIdx.y+blockIdx.y*blockDim.y;

if(i<height+bottom &&j<width){

float angle = Dir[i*width+j];

if(angle<0) angle = angle + M_PI;

angle=(180/M_PI)*angle;

// top and bottom
if(angle > 157.5 || angle <= 22.5){
if (i-1 >= 0-top && i+1 < height+bottom) {
if (Mag[(i-1)*width+j]>Mag[i*width+j] || Mag[(i+1)*width+j]>Mag[i*width+j])
sup[i*width+j]=0;

}

}
// top left and right botom
else if (angle>22.5 && angle<=67.5) {
if ( (i-1) >= 0-top && (j-1) >= 0){
if (Mag[(i-1)*width+(j-1)] > Mag[i*width+j]){
sup[i*width+j]=0;
}else if((i+1<height+bottom && j+1 <width)){
if(Mag[(i+1)*width+(j+1)]>Mag[i*width+j])
sup[i*width+j]=0;

}}}


//left and right
else if(angle>67.5 && angle<=112.5){
if (j-1 >= 0 && j+1 < width) {
if (Mag[i*width+(j-1)]>Mag[i*width+j] || Mag[i*width+(j+1)]>Mag[i*width+j]) {
sup[i*width+j]=0;
}
}
}
// left bottom and right top
else if(angle>112.5 && angle<=157.5){
if ((j-1 >= 0 && i-1 >= 0-top ) &&(i+1 < height+bottom && j+1 < width)) {
if (Mag[(i+1)*width+(j-1)]>Mag[i*width+j] || Mag[(i-1)*width+(j+1)]>Mag[i*width+j]) {
sup[i*width+j]=0;
}
}
}

}

}

void supression(float *d_sup, float *Mag, float *Dir, int width, int height, int comm_rank, int comm_size, int blocksize, int a){

	float *d_Mag,*d_Dir;
        cudaMalloc((void **)&d_Dir,sizeof(float)*width*height);
        cudaMalloc((void **)&d_sup,sizeof(float)*width*height);

	if(comm_rank == 0 || comm_rank == (comm_size-1))
	cudaMalloc((void **)&d_Mag,sizeof(float)*width*(height+a));
	
	else
	cudaMalloc((void **)&d_Mag,sizeof(float)*width*(height+2*a));
	


    	cudaMemcpy(d_Dir,Dir,sizeof(float)*width*height,cudaMemcpyHostToDevice);
     	cudaMemcpy(d_sup,Mag,sizeof(float)*width*height,cudaMemcpyHostToDevice);
        
	if(comm_rank == 0 || comm_rank == (comm_size-1))
	cudaMemcpy(d_Mag,Mag,sizeof(float)*width*(height+a),cudaMemcpyHostToDevice);
	else
	cudaMemcpy(d_Mag,Mag,sizeof(float)*width*height+2*a,cudaMemcpyHostToDevice);

	
    	int top, bottom;
    
   	if (comm_rank == 0){
        	top = 0;
        	bottom = 1;
    	}else if (comm_rank == comm_size-1){
        	top = 1;
        	bottom = 0;
    	}else{
        	top = 1;
        	bottom = 1;
    	}

        dim3 dimBlock(blocksize,blocksize,1);
        dim3 dimGrid(ceil(height/blocksize),ceil(width/blocksize),1);

	supressionGPU<<<dimGrid,dimBlock>>>(d_sup, d_Mag, d_Dir, height, width, top, bottom);

       // cudaMemcpy(sup,d_sup,sizeof(float)*width*height,cudaMemcpyDeviceToHost);


        cudaFree(d_Dir);
        cudaFree(d_Mag);
        cudaFree(d_sup);

}






__global__
void hysteresisGPU(float *sup, float *hys, int height, int width, float t_high, float t_low){

int i,j;
i=threadIdx.x+blockIdx.x*blockDim.x;
j=threadIdx.y+blockIdx.y*blockDim.y;


if(i<height && j <width){
if(sup[i*width+j]>=t_high)
hys[i*width+j]=255;
else if(sup[i*width+j]<=t_low)
hys[i*width+j]=0;
else if(sup[i*width+j]<t_high && sup[i*width+j]>t_low)
hys[i*width+j]=125;
}
}


void hysteresis(float *sup, float *d_hys, int height, int width, float t_high, float t_low, int blocksize){
   

	float *d_sup;
        cudaMalloc((void **)&d_sup,sizeof(float)*width*height);
        cudaMalloc((void **)&d_hys,sizeof(float)*width*height);



    	cudaMemcpy(d_sup,sup,sizeof(float)*width*height,cudaMemcpyHostToDevice);
     	cudaMemcpy(d_hys,sup,sizeof(float)*width*height,cudaMemcpyHostToDevice);
        
        dim3 dimBlock(blocksize,blocksize,1);
        dim3 dimGrid(ceil(height/blocksize),ceil(width/blocksize),1);

	hysteresisGPU<<<dimGrid,dimBlock>>>(d_sup, d_hys, height, width, t_high, t_low);

       // cudaMemcpy(hys,d_hys,sizeof(float)*width*height,cudaMemcpyDeviceToHost);


        cudaFree(d_hys);
        cudaFree(d_sup);



}



__global__
void FinaledgeGPU(float *edge, float *hys, int height, int width, int top, int bottom ){
int i,j;
i=threadIdx.x+blockIdx.x*blockDim.x;
j=threadIdx.y+blockIdx.y*blockDim.y;
//edge[i*width+j]=hys[i*width+j];

for (int y=-1; y<=1; y++){
for (int x=-1; x<=1; x++){
if(i+y<height+bottom && i+y>0-top && j+x<width && j+x> 0){
if (hys[(i+y)*width+x+j]==255)

edge[i*width+j]=255;

else

edge[i*width+j]=0;
}
}
}
}


void Finaledge(float *d_edge, float *hys, int height, int width, int comm_size, int comm_rank, int blocksize){
    
    
    int top, bottom;
    
    if (comm_rank == 0){
        top = 0;
        bottom = 1;
    }else if (comm_rank == comm_size-1){
        top = 1;
        bottom = 0;
    }else{
        top = 1;
        bottom = 1;
    }

	float *d_hys;
        cudaMalloc((void **)&d_edge,sizeof(float)*width*height);
        cudaMalloc((void **)&d_hys,sizeof(float)*width*height);



    	cudaMemcpy(d_edge,hys,sizeof(float)*width*height,cudaMemcpyHostToDevice);
     	cudaMemcpy(d_hys,hys,sizeof(float)*width*height,cudaMemcpyHostToDevice);
        
        dim3 dimBlock(blocksize,blocksize,1);
        dim3 dimGrid(ceil(height/blocksize),ceil(width/blocksize),1);

	FinaledgeGPU<<<dimGrid,dimBlock>>>(d_edge, d_hys, height, width, top, bottom);

       // cudaMemcpy(edge,d_edge,sizeof(float)*width*height,cudaMemcpyDeviceToHost);


        cudaFree(d_edge);
        cudaFree(d_hys);



   
}



__global__
void feature_detecGPU(float *d_cornerness,int height, int width, float *C_ver, float *C_hor, int blocksize, int top, int bottom){

//float k = 0.04;
int window_width = 7;
//float *cornerness = (float*)malloc(sizeof(float)*height*width);
//float *C_hor = (float*)malloc(sizeof(float)*height*width);
//float *C_ver = (float*)malloc(sizeof(float)*height*width);
int locaIx = threadIdx.x;
int locaIy = threadIdx.y;
int globIx = threadIdx.x+blockIdx.x*blockDim.x;
int globIy = threadIdx.y+blockIdx.y*blockDim.y;
float Ixx,Iyy,IxIy;

extern __shared__ float Ashared[];
__shared__ float *Vshared, *Hshared;
Vshared = Ashared;
Hshared = Ashared+blocksize*blocksize;

Vshared[locaIx*blockDim.y+locaIy] = C_ver[globIx*width+globIy];
Hshared[locaIx*blockDim.y+locaIy] = C_hor[globIx*width+globIy];
__syncthreads();
Ixx = 0;
Iyy = 0;
IxIy = 0;

if(globIx <height+bottom && globIy < width){
for(int k = -window_width/2; k < window_width/2 ; k++){
for(int m = -window_width/2; m < window_width/2 ; m++){

if(locaIx+k >= 0 && locaIx+k < blockDim.x && locaIy+m >= 0 && locaIy+m < blockDim.y){
int offseti = locaIx+k;
int offsetj = locaIy+m;

Ixx = Ixx + pow(Vshared[offseti*blockDim.y+offsetj],2);
Iyy = Iyy + pow(Hshared[offseti*blockDim.y+offsetj],2);
IxIy = IxIy + Vshared[offseti*blockDim.y+offsetj] * Hshared[offseti*blockDim.y+offsetj];

}

else if(globIx+k >= 0-top && globIx+k < height+bottom && globIy+m >= 0 && globIy+m < width){
int offseti = globIx+k;
int offsetj = globIy+m;

Ixx = Ixx + pow(C_ver[offseti*width+offsetj],2);
Iyy = Iyy + pow(C_hor[offseti*width+offsetj],2);
IxIy = IxIy + C_ver[offseti*width+offsetj] * C_hor[offseti*width+offsetj];

}
}
}
__syncthreads();
d_cornerness[globIx*width+globIy]= (Ixx*Iyy) - (IxIy*IxIy) - 0.04*(Ixx+Iyy)*(Ixx+Iyy);

printf("test");
}
}


void feature_detec(float *feature, int height, int width, float *vertical, float *horizon, int comm_size, int comm_rank, int blocksize, int a){
	float *d_ver,*d_hor,*d_feature;
	    
    int top, bottom;
    
    if (comm_rank == 0){
        top = 0;
        bottom = 1;
    }else if (comm_rank == comm_size-1){
        top = 1;
        bottom = 0;
    }else{
        top = 1;
        bottom = 1;
    }

       cudaMalloc((void **)&d_feature,sizeof(float)*width*height);
	  
	if(comm_rank == 0 || comm_rank == (comm_size-1)){
	cudaMalloc((void **)&d_ver,sizeof(float)*width*(height+a));
        cudaMalloc((void **)&d_hor,sizeof(float)*width*(height+a));
	}else{
	cudaMalloc((void **)&d_ver,sizeof(float)*width*(height+2*a));
        cudaMalloc((void **)&d_hor,sizeof(float)*width*(height+2*a));
	}

	if(comm_rank == 0 || comm_rank == (comm_size-1)){
	cudaMemcpy(d_ver,vertical,sizeof(float)*width*(height+a),cudaMemcpyHostToDevice);
	cudaMemcpy(d_hor,horizon,sizeof(float)*width*(height+a),cudaMemcpyHostToDevice);
	
	}else{
	cudaMemcpy(d_ver,vertical,sizeof(float)*width*(height+2*a),cudaMemcpyHostToDevice);	
	cudaMemcpy(d_hor,horizon,sizeof(float)*width*(height+2*a),cudaMemcpyHostToDevice);
}

        
        dim3 dimBlock(blocksize,blocksize,1);
        dim3 dimGrid(ceil(height/blocksize),ceil(width/blocksize),1);

	feature_detecGPU<<<dimGrid,dimBlock>>>(d_feature, height, width, d_ver, d_hor, blocksize, top , bottom);
        cudaMemcpy(feature,d_feature,sizeof(float)*width*height,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();



        cudaFree(d_ver);
        cudaFree(d_hor);
	cudaFree(d_feature);





}



__global__
void find_featureGPU(float *output, float *d_cornerness,int height, int width, int blocksize){

	printf("here\n");
//int locatI,locatJ;
int stride,localIndex;
//    struct location *loc;
int window_size = blockDim.x;
//int window_width = blockDim.y;
int locaIx = threadIdx.x;
int locaIy = threadIdx.y;
//int Index_locaIx = blocksize*blocksize+threadIdx.x;
//int Index_locaIy = blocksize*blocksize+threadIdx.y;
int globIx = threadIdx.x+blockIdx.x*blockDim.x;
int globIy = threadIdx.y+blockIdx.y*blockDim.y;
extern __shared__ float Shared[];
__shared__ float *AShared,*indexShared;
AShared = Shared;
indexShared = Shared+blocksize*blocksize;

//float kerw=(k_width>k_height)?k_width:k_height;

AShared[locaIx*blockDim.y+locaIy] = d_cornerness[globIx*width+globIy];
indexShared[locaIx*blockDim.y+locaIy] = globIx*width+globIy;
//int a = indexShared[locaIx*blockDim.y+locaIy];
//printf("a; %d ",a);
__syncthreads();
// loc = (struct location*)malloc(sizeof(struct location)*window_heigh*window_width);
// int locount = 0;
//printf("%d ",AShared+locaIx*blockDim.y+locaIy);
// if(globIx < height && globIy <width){
for (stride = ((window_size*window_size)/2);stride >= 1; stride/=2){
__syncthreads();
localIndex = locaIx*blockDim.y+locaIy;
if(localIndex < stride){
if(AShared[localIndex]<AShared[localIndex+stride]){
AShared[localIndex]=AShared[localIndex+stride];
indexShared[localIndex]=indexShared[localIndex+stride];

//    }else if(AShared[localIndex]<AShared[localIndex+stride])

//        AShared[localIndex]=AShared[localIndex+stride];
//                indexShared[localIndex]=localIndex+stride ;
}
}
}

if(locaIx == 0 && locaIy == 0){
output[globIx*width+globIy]=indexShared[0];

int a = indexShared[0];
printf("%d",a);
}
}


void find_feature(float *output,float *temp_feature,int height,int width,int comm_size, int comm_rank,int blocksize){

	float *d_temp_feature, *d_output;
        
	cudaMalloc((void **)&d_output,sizeof(float)*width*height);
        cudaMalloc((void **)&d_temp_feature,sizeof(float)*width*height);



     	cudaMemcpy(d_temp_feature,temp_feature,sizeof(float)*width*height,cudaMemcpyHostToDevice);
        
        dim3 dimBlock(blocksize,blocksize,1);
        dim3 dimGrid(ceil(height/blocksize),ceil(width/blocksize),1);

	find_featureGPU<<<dimGrid,dimBlock,2*sizeof(float)*blocksize*blocksize>>>(d_output, d_temp_feature, height, width,blocksize);

        cudaMemcpy(output,d_output,sizeof(float)*width*height,cudaMemcpyDeviceToHost);

        //cudaFree(d_output);
       // cudaFree(d_temp_feature);


}





/*




int main(int argc, char ** argv){

int blocksize = atof(argv[3]);
int height,width, w;
float *image,sigma, a;
float *gauss_Kernel,*dgauss_Kernel;
//float *horizon;
//    float *vertical;
float *Mag,*sup,*Dir,*hys;
float *edge;
//create pointers for GPU
cudaSetDevice(0); //use GPU 0
float *d_image;
float *d_gauss_Kernel,*d_dgauss_Kernel;
float *d_temp_horizon,*d_horizon;
float *d_vertical, *d_temp_vertical;
float *d_Mag,*d_Dir,*d_sup, *d_hys, *d_edge;
float *d_cornerness,*cornerness,*d_features,*features;
struct timeval fileIStart,fileIEnd,fileOStart,fileOEnd,k1Start,k1End,k2Start,k2End,k3Start,k3End,k4Start,k4End,k5Start,k5End,k6Start,k6End,k7Start,k7End,k8Start,k8End,k9Start,k9End,H2DStart,H2DEnd,D2HStart,D2HEnd,start, end, computationstart,computationend;

gettimeofday(&start, NULL);

//for file input timer
gettimeofday(&fileIStart,NULL);
read_image_template(argv[1],&image,&width,&height);
gettimeofday(&fileIEnd, NULL);

gettimeofday(&computationstart, NULL);

sigma = atof(argv[2]);
a = round(2.5*sigma-0.5);
w = 2*a+1;
//printf("a:%f w:%d sigma: %f \n",a,w,sigma);
//Malloc for CPU
gauss_Kernel=(float*)malloc(sizeof(float)*w);
dgauss_Kernel=(float*)malloc(sizeof(float)*w);
//temp_horizon=(float *)malloc(sizeof(float)*width*height);
//horizon=(float *)malloc(sizeof(float)*width*height);
//temp_vertical=(float *)malloc(sizeof(float)*width*height);
//vertical=(float *)malloc(sizeof(float)*width*height);
//Mag=(float *)malloc(sizeof(float)*width*height);
//Dir=(float *)malloc(sizeof(float)*width*height);
sup=(float *)malloc(sizeof(float)*width*height);
//hys=(float *)malloc(sizeof(float)*width*height);
edge=(float *)malloc(sizeof(float)*width*height);
cornerness = (float*)malloc(sizeof(float)*height*width);
features = (float*)malloc(sizeof(float)*height*width);
//Malloc for GPU
cudaMalloc((void **)&d_image,sizeof(float)*width*height);
cudaMalloc((void **)&d_gauss_Kernel,sizeof(float)*w);
cudaMalloc((void **)&d_dgauss_Kernel,sizeof(float)*w);
cudaMalloc((void **)&d_temp_horizon,sizeof(float)*width*height);
cudaMalloc((void **)&d_horizon,sizeof(float)*width*height);
cudaMalloc((void **)&d_temp_vertical,sizeof(float)*width*height);
cudaMalloc((void **)&d_vertical,sizeof(float)*width*height);
cudaMalloc((void **)&d_Mag,sizeof(float)*width*height);
cudaMalloc((void **)&d_Dir,sizeof(float)*width*height);
cudaMalloc((void **)&d_sup,sizeof(float)*width*height);
cudaMalloc((void **)&d_hys,sizeof(float)*width*height);
cudaMalloc((void **)&d_edge,sizeof(float)*width*height);
cudaMalloc((void **)&d_cornerness,sizeof(float)*width*height);
cudaMalloc((void **)&d_features,sizeof(float)*width*height);


Cal_gauss_kernel(gauss_Kernel,sigma,a, w, dgauss_Kernel);
//printf("Gaussian Kernel:\n");
//print_matrix(gauss_Kernel, 1, w);
//printf("Derivative Kernel:\n");
//print_matrix(dgauss_Kernel,1,w);
//copy data from CPU to GPU

gettimeofday(&H2DStart, NULL);

cudaMemcpy(d_image,image,sizeof(float)*width*height,cudaMemcpyHostToDevice);
cudaMemcpy(d_gauss_Kernel,gauss_Kernel,sizeof(float)*w,cudaMemcpyHostToDevice);
cudaMemcpy(d_dgauss_Kernel,dgauss_Kernel,sizeof(float)*w,cudaMemcpyHostToDevice);
cudaDeviceSynchronize();
gettimeofday(&H2DEnd, NULL);

//Horizonal gradient
//int blocksize = atof(argv[3]);
dim3 dimBlock(blocksize,blocksize,1);
dim3 dimGrid(ceil(height/blocksize),ceil(width/blocksize),1);

gettimeofday(&k1Start, NULL);

convolveGPU<<<dimGrid,dimBlock,sizeof(float)*blocksize*blocksize>>>(d_image, d_temp_horizon, d_gauss_Kernel, height, width, w, 1 );
convolveGPU<<<dimGrid,dimBlock,sizeof(float)*blocksize*blocksize>>>(d_temp_horizon, d_horizon, d_dgauss_Kernel, height, width, 1, w);
cudaDeviceSynchronize();

gettimeofday(&k1End, NULL);

//convolve(image, &temp_horizon, gauss_Kernel, height, width, w, 1 );
//convolve(temp_horizon, &horizon, dgauss_Kernel, height, width, 1, w);

//cudaMemcpy(horizon,d_horizon,sizeof(float)*width*height,cudaMemcpyDeviceToHost);

//Vertical gradient

//convolve(image, &temp_vertical, gauss_Kernel, height, width,1, k_w);
//convolve(temp_vertical, &vertical, dgauss_Kernel, height, width, k_w, 1);

gettimeofday(&k2Start, NULL);
convolveGPU<<<dimGrid,dimBlock,sizeof(float)*blocksize*blocksize>>>(d_image, d_temp_vertical, d_gauss_Kernel, height, width, 1,w );
convolveGPU<<<dimGrid,dimBlock,sizeof(float)*blocksize*blocksize>>>(d_temp_horizon, d_vertical, d_dgauss_Kernel, height, width, w,1);
cudaDeviceSynchronize();
gettimeofday(&k2End, NULL);
//cudaMemcpy(vertical, d_vertical,sizeof(float)*width*height,cudaMemcpyDeviceToHost);

// Magnitude
gettimeofday(&k3Start, NULL);
//Magnitude(vertical, horizon, &Mag, height, width);
MagnitudeGPU<<<dimGrid,dimBlock>>>(d_vertical, d_horizon, d_Mag, height, width);
cudaDeviceSynchronize();
gettimeofday(&k3End, NULL);
//cudaMemcpy(Mag, d_Mag, sizeof(float)*width*height,cudaMemcpyDeviceToHost);
// Direction
gettimeofday(&k4Start, NULL);
//Direction(vertical, horizon, &Dir, height, width);
DirectionGPU<<<dimGrid,dimBlock>>>(d_vertical, d_horizon, d_Dir, height, width);
cudaDeviceSynchronize();
gettimeofday(&k4End, NULL);
//cudaMemcpy(Dir, d_Dir, sizeof(float)*width*height,cudaMemcpyDeviceToHost);
// supression
//supression (&sup, Mag, Dir, height, width);
gettimeofday(&k5Start, NULL);
supressionGPU<<<dimGrid,dimBlock>>>(d_sup, d_Mag, d_Dir, height, width);
cudaDeviceSynchronize();
gettimeofday(&k5End, NULL);
//cudaMemcpy(sup, d_sup, sizeof(float)*width*height,cudaMemcpyDeviceToHost);


// hysteresis
thrust::device_ptr<float>thr_d(d_sup);
thrust::device_vector<float>d_sup_vec(thr_d,thr_d+(height*width));
thrust::sort(d_sup_vec.begin(),d_sup_vec.end());
int index = (int)((0.9)*height*width);
float t_high = d_sup_vec[index];
float t_low = t_high/5;

gettimeofday(&k6Start, NULL);
hysteresisGPU<<<dimGrid,dimBlock>>>(d_sup, d_hys, height, width, t_high, t_low);
cudaDeviceSynchronize();
//cudaMemcpy(hys,d_hys,sizeof(float)*width*height,cudaMemcpyDeviceToHost);
gettimeofday(&k6End, NULL);
// Finaledge
gettimeofday(&k7Start, NULL);
FinaledgeGPU<<<dimGrid,dimBlock>>>(d_edge, d_hys, height, width);
cudaDeviceSynchronize();
gettimeofday(&k7End, NULL);


gettimeofday(&D2HStart, NULL);
cudaMemcpy(edge,d_edge,sizeof(float)*width*height,cudaMemcpyDeviceToHost);
gettimeofday(&D2HEnd, NULL);

// feature
gettimeofday(&k8Start, NULL);
feature_detecGPU<<<dimGrid,dimBlock,2*sizeof(float)*blocksize*blocksize>>>(d_cornerness,height, width, d_vertical, d_horizon, blocksize);

cudaDeviceSynchronize();
gettimeofday(&k8End, NULL);
//cudaMemcpy(cornerness,d_cornerness,sizeof(float)*width*height,cudaMemcpyDeviceToHost);
//    gettimeofday(&k9Start, NULL);
gettimeofday(&k9Start, NULL);
find_featureGPU<<<dimGrid,dimBlock,2*sizeof(float)*blocksize*blocksize>>>(d_features,d_cornerness,height, width,blocksize);
cudaDeviceSynchronize();
gettimeofday(&k9End, NULL);

cudaMemcpy(features,d_features,sizeof(float)*width*height,cudaMemcpyDeviceToHost);
gettimeofday(&computationend, NULL);



//    	FILE  *T0;
//    	T0=fopen("index.csv","w+");
	int location_I, location_J;

	for(int i = 0; i<width*height; i++){
	
	if (features[i]>0){
	int a = *(features+i);
	location_I = a/width;
	location_J = a%width;
      
	printf("Index:%d, I:%d, J:%d\n",a,location_I,location_J);
	
	}
}
//	fclose(T0);


//output image
//write_image_template("h_convolve.pgm",horizon, width, height);
//write_image_template("v_convolve.pgm",vertical, width, height);
//write_image_template("Magnitude.pgm",Mag, width, height);
//write_image_template("Direction.pgm",Dir, width, height);
//write_image_template("suppress.pgm", sup, width, height);
//write_image_template("hysteresis.pgm", hys, width, height);


gettimeofday(&fileOStart,NULL);
write_image_template("edge.pgm", edge, width, height);
gettimeofday(&fileOEnd, NULL);
//write_image_template("cornerness.pgm", cornerness, width, height);

//free
//free(Mag);
//free(Dir);
//free(horizon);
//free(vertical);
//free(sup);
//free(hys);
free(edge);
cudaFree(d_image);
//    cudaFree(d_image);
cudaFree(d_temp_horizon);
cudaFree(d_horizon);
cudaFree(d_temp_vertical);
cudaFree(d_vertical);
cudaFree(d_Mag);
cudaFree(d_Dir);
cudaFree(d_sup);
cudaFree(d_hys);
cudaFree(d_edge);

gettimeofday(&end, NULL);


printf("BlockSize: %d Image-Height: %d Width: %d Sigma: %f file i/o time: %ld kernel time: %ld communication time: %ld end to end with i/o: %ld end to end with no i/o: %ld\n",blocksize,height,width,atof(argv[2]),(((fileOEnd.tv_sec *1000000 + fileOEnd.tv_usec)-(fileOStart.tv_sec * 1000000 + fileOStart.tv_usec))+((fileIEnd.tv_sec *1000000 + fileIEnd.tv_usec)-(fileIStart.tv_sec * 1000000 + fileIStart.tv_usec))),(((k1End.tv_sec *1000000 + k1End.tv_usec)-(k1Start.tv_sec * 1000000 + k1Start.tv_usec))+((k2End.tv_sec *1000000 + k2End.tv_usec)-(k2Start.tv_sec * 1000000 + k2Start.tv_usec))+((k3End.tv_sec *1000000 + k3End.tv_usec)-(k3Start.tv_sec * 1000000 + k3Start.tv_usec))+((k4End.tv_sec *1000000 + k4End.tv_usec)-(k4Start.tv_sec * 1000000 + k4Start.tv_usec))+((k5End.tv_sec *1000000 + k5End.tv_usec)-(k5Start.tv_sec * 1000000 + k5Start.tv_usec))+((k6End.tv_sec *1000000 + k6End.tv_usec)-(k6Start.tv_sec * 1000000 + k6Start.tv_usec))+((k7End.tv_sec *1000000 + k7End.tv_usec)-(k7Start.tv_sec * 1000000 + k7Start.tv_usec))),((H2DEnd.tv_sec *1000000 + H2DEnd.tv_usec)-(H2DStart.tv_sec * 1000000 + H2DStart.tv_usec)+(D2HEnd.tv_sec *1000000 + D2HEnd.tv_usec)-(D2HStart.tv_sec * 1000000 + D2HStart.tv_usec)),(end.tv_sec *1000000 + end.tv_usec)-(start.tv_sec * 1000000 + start.tv_usec),
(computationend.tv_sec *1000000 + computationend.tv_usec)-(computationstart.tv_sec * 1000000 + computationstart.tv_usec));

}*/




               
