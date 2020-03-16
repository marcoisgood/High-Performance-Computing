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

__global__
void convolveGPU(float *image, float *output, float *kernel, int height,int width, int k_height, int k_width, int top, int bottom){

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
output[i*width+j]=sum;
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





void convolve(float *image, float *output, float *kernel, int height,
              int width, int k_height, int k_width, int comm_size, int comm_rank, int blocksize){
    
  	int w = k_height>1?k_height:k_width;
	int a = floor(w/2);

	float *d_image,*d_temp_horizon,*d_Kernel;
	

	if(comm_rank == 0 || comm_rank == (comm_size-1))
	cudaMalloc((void **)&d_image,sizeof(float)*width*(height+a));
	
	else
	cudaMalloc((void **)&d_image,sizeof(float)*width*(height+2*a));


	cudaMalloc((void **)&d_Kernel,sizeof(float)*w);
	cudaMalloc((void **)&d_temp_horizon,sizeof(float)*width*height);



	if(comm_rank == 0 || comm_rank == (comm_size-1))
	cudaMemcpy(d_image,image,sizeof(float)*width*(height+a),cudaMemcpyHostToDevice);
	else
	cudaMemcpy(d_image,image,sizeof(float)*width*(height+2*a),cudaMemcpyHostToDevice);

	cudaMemcpy(d_image,image,sizeof(float)*width*height,cudaMemcpyHostToDevice);


	cudaMemcpy(d_Kernel,kernel,sizeof(float)*w,cudaMemcpyHostToDevice);

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
 	
	convolveGPU<<<dimGrid,dimBlock,sizeof(float)*blocksize*blocksize>>>(d_image, d_temp_horizon, d_Kernel, height, width, k_height, k_width, top, bottom);   
       
	cudaMemcpy(output,d_temp_horizon,sizeof(float)*width*height,cudaMemcpyDeviceToHost);
	    
	cudaFree(d_image);
	cudaFree(d_temp_horizon);
	cudaFree(d_Kernel);

}








__global__
void MagnitudeGPU(float *vertical, float *horizon, float *Mag, int height, int width){

int i,j;
i=threadIdx.x+blockIdx.x*blockDim.x;
j=threadIdx.y+blockIdx.y*blockDim.y;

if(i<height && j<width)
Mag[i*width+j]=sqrt(pow(vertical[i*width+j],2)+pow(horizon[i*width+j],2));

}


void Magnitude(float *vertical, float *horizon,float *Mag, int height, int width, int blocksize){
	

	float *d_Mag,*d_vertical,*d_horizon;
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

 
	cudaMemcpy(Mag,d_Mag,sizeof(float)*width*height,cudaMemcpyDeviceToHost);
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


void Direction(float *vertical, float *horizon, float *Dir, int height,int width, int blocksize){
    
 
	float *d_Dir,*d_vertical,*d_horizon;
	cudaMalloc((void **)&d_Dir,sizeof(float)*width*height);
 	cudaMalloc((void **)&d_vertical,sizeof(float)*width*height);  
 	cudaMalloc((void **)&d_horizon,sizeof(float)*width*height);
	
	cudaMemcpy(d_vertical,vertical,sizeof(float)*width*height,cudaMemcpyHostToDevice);
	cudaMemcpy(d_horizon,horizon,sizeof(float)*width*height,cudaMemcpyHostToDevice);
	
	dim3 dimBlock(blocksize,blocksize,1);
	dim3 dimGrid(ceil(height/blocksize),ceil(width/blocksize),1);


	DirectionGPU<<<dimGrid,dimBlock>>>(d_vertical, d_horizon, d_Dir, height, width);   
  
	cudaMemcpy(Dir,d_Dir,sizeof(float)*width*height,cudaMemcpyDeviceToHost);
   
       
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

void supression(float *sup, float *Mag, float *Dir, int width, int height, int comm_rank, int comm_size, int blocksize, int a){

	float *d_sup,*d_Mag,*d_Dir;
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

        cudaMemcpy(sup,d_sup,sizeof(float)*width*height,cudaMemcpyDeviceToHost);


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


void hysteresis(float *sup, float *hys, int height, int width, float t_high, float t_low, int blocksize){
   

	float *d_sup,*d_hys;
        cudaMalloc((void **)&d_sup,sizeof(float)*width*height);
        cudaMalloc((void **)&d_hys,sizeof(float)*width*height);



    	cudaMemcpy(d_sup,sup,sizeof(float)*width*height,cudaMemcpyHostToDevice);
     	cudaMemcpy(d_hys,sup,sizeof(float)*width*height,cudaMemcpyHostToDevice);
        
        dim3 dimBlock(blocksize,blocksize,1);
        dim3 dimGrid(ceil(height/blocksize),ceil(width/blocksize),1);

	hysteresisGPU<<<dimGrid,dimBlock>>>(d_sup, d_hys, height, width, t_high, t_low);

        cudaMemcpy(hys,d_hys,sizeof(float)*width*height,cudaMemcpyDeviceToHost);


        cudaFree(d_hys);
        cudaFree(d_sup);



}



__global__
void FinaledgeGPU(float *edge, float *hys, int height, int width, int top, int bottom ){
int i,j;
i=threadIdx.x+blockIdx.x*blockDim.x;
j=threadIdx.y+blockIdx.y*blockDim.y;
edge[i*width+j]=hys[i*width+j];

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


void Finaledge(float *edge, float *hys, int height, int width, int comm_size, int comm_rank, int blocksize){
    
    
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

	float *d_edge,*d_hys;
        cudaMalloc((void **)&d_edge,sizeof(float)*width*height);
        cudaMalloc((void **)&d_hys,sizeof(float)*width*height);



    	cudaMemcpy(d_edge,hys,sizeof(float)*width*height,cudaMemcpyHostToDevice);
     	cudaMemcpy(d_hys,hys,sizeof(float)*width*height,cudaMemcpyHostToDevice);
        
        dim3 dimBlock(blocksize,blocksize,1);
        dim3 dimGrid(ceil(height/blocksize),ceil(width/blocksize),1);

	FinaledgeGPU<<<dimGrid,dimBlock>>>(d_edge, d_hys, height, width, top, bottom);

        cudaMemcpy(edge,d_edge,sizeof(float)*width*height,cudaMemcpyDeviceToHost);


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


/*
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

	find_featureGPU<<<dimGrid,dimBlock,2*sizeof(float)*blocksize*blocksize >>>(d_output, d_temp_feature, height, width,blocksize);
	
        cudaMemcpy(output,d_output,sizeof(float)*width*height,cudaMemcpyDeviceToHost);

        //cudaFree(d_output);
       // cudaFree(d_temp_feature);


}


*/




               
