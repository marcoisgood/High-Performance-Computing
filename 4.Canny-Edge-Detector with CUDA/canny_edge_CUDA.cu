#include<stdio.h>
#include <string.h>
#include "stdlib.h"
#include<math.h>
#include"image_template.h"
#include"time.h"
#include"sys/time.h"
#include<cuda.h>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include<thrust/sort.h>
#include<thrust/copy.h>
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

/* void convolve(float *image, float **output, float *kernel, int height,
int width, int k_height, int k_width){

*output = (float*)malloc(sizeof(float)*height*width);

for(int i=0; i<height; i++){
for(int j=0; j<width; j++){
float sum = 0;
for(int k=0; k<k_height;k++){
for(int m=0; m<k_width; m++){
int offseti = -1*floor(k_height/2)+k;
int offsetj = -1*floor(k_width/2)+m;

if(range(j+offsetj, i+offseti, height, width)){
sum+= *(image+(i+offseti)*width+j+offsetj)*(*(kernel+(k*k_width)+m));
}
}
}
*(*output+(i*width)+j)=sum;

}
}
}*/

__global__
void convolveGPU(float *image, float *output, float *kernel, int height,int width, int k_height, int k_width){


/*

//using global memory

int i,j,k,m,offseti,offsetj;
float kerw=(k_width>k_height)?k_width:k_height;
//printf("%f",kerw);
i=threadIdx.x+blockIdx.x*blockDim.x;
j=threadIdx.y+blockIdx.y*blockDim.y;

if(i<height && j<width ){
float sum = 0;
for( m=0; m<kerw; m++){
offseti = k_height>1?(-1*(k_height/2)+m):0;
offsetj = k_width>1?(-1*(k_width/2)+m):0;
if( (i+offseti)>=0 && (i+offseti)<height && (j+offsetj)>=0 && (j+offsetj)< width)
sum+= image[(i+offseti)*width+(j+offsetj)]*kernel[m];
}
output[i*width+j]=sum;
}*/




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
}






void Cal_gauss_kernel(float *gauss_Kernel, float sigma, float pw,int w, float *dgauss_Kernel){
// float pw = round(2.5*sigma-0.5);
float sum = 0;
//printf("a:%f w:%d sigma: %f \n",pw,w,sigma);
// *w = 2*pw+1;

// *gauss_Kernel=(float*)malloc(sizeof(float)*(*w));
for (int i= 0; i<w; i++){
gauss_Kernel[i] = exp((-1*(i-pw)*(i-pw))/(2*sigma*sigma));
sum += gauss_Kernel[i];
}
for (int i=0; i<w; i++){
gauss_Kernel[i] = gauss_Kernel[i]/sum;
}

sum = 0;

// *dgauss_Kernel=(float*)malloc(sizeof(float)*(*w));

for(int i=0; i<w; i++){
dgauss_Kernel[i] = -1*(i-pw)*exp((-1*(i-pw)*(i-pw))/(2*sigma*sigma));
sum-=i*(dgauss_Kernel[i]);
}

for(int i=0; i<w; i++){
dgauss_Kernel[i]/=sum;
}
float temp;
int start=0, end= w-1;

while (start < end){
temp=dgauss_Kernel[start];
dgauss_Kernel[start]=dgauss_Kernel[end];
dgauss_Kernel[end]=temp;
start++;
end--;
}
}


/*    void Magnitude(float *vertical, float *horizon,float **Mag, int height, int width){

*Mag = (float*)malloc(sizeof(float)*height*width);

for(int i=0; i<height; i++){
for(int j=0; j<width; j++){


*(*Mag+i*width+j)=sqrt(pow(*(vertical+i*width+j),2)
+ pow(*(horizon+i*width+j),2));
}
}

}
*/
__global__
void MagnitudeGPU(float *vertical, float *horizon, float *Mag, int height, int width){

int i,j;
i=threadIdx.x+blockIdx.x*blockDim.x;
j=threadIdx.y+blockIdx.y*blockDim.y;

if(i<height && j<width)
Mag[i*width+j]=sqrt(pow(vertical[i*width+j],2)+pow(horizon[i*width+j],2));

}

/*
void Direction(float *vertical, float *horizon, float **Dir, int height,int width){

*Dir = (float*)malloc(sizeof(float)*height*width);

for(int i=0; i<height; i++){
for(int j=0; j<width; j++){


*(*Dir+i*width+j)=atan2(*(vertical+i*width+j),*(horizon+i*width+j));
}
}
}*/
__global__
void DirectionGPU(float *vertical, float *horizon, float *Dir, int height,int width){

int i,j;
i=threadIdx.x+blockIdx.x*blockDim.x;
j=threadIdx.y+blockIdx.y*blockDim.y;
if(i<height&&j<width)
Dir[i*width+j]=atan2(vertical[i*width+j],horizon[i*width+j]);


}

/*
void supression(float **sup, float *Mag, float *Dir, int width, int height){
*sup = (float*)malloc(sizeof(float)*height*width);
memcpy(*sup, Mag, sizeof(float)*height*width);

for (int i =0 ; i< height; i++){
for (int j = 0; j< width; j++){

float angle = *(Dir+i*width+j);

if(angle<0) angle = angle + M_PI;

angle=(180/M_PI)*angle;

// top and bottom
if(angle > 157.5 && angle <= 22.5){
if (i-1 >= 0 && i+1 < height) {
if (*(Mag+(i-1)*width+j)>*(Mag+i*width+j) || *(Mag+(i+1)*width+j)>*(Mag+i*width+j)){
*(*sup+i*width+j)=0;
}
}

}
// top left and right botom
else if (angle>22.5 && angle<=67.5) {
if ( (i-1 >= 0 && j-1 >= 0) || ( i+1 < height && j+1 < width)) {
if (*(Mag+(i-1)*width+(j-1))>*(Mag+i*width+j) || *(Mag+(i+1)*width+(j+1))>*(Mag+i*width+j)) {
*(*sup+i*width+j)=0;
}

}

}
//left and right
else if(angle>67.5 && angle<=112.5){
if (j-1 >= 0 && j+1 < width) {
if (*(Mag+i*width+(j-1))>*(Mag+i*width+j) || *(Mag+i*width+(j+1))>*(Mag+i*width+j)) {
*(*sup+i*width+j)=0;
}
}
}
// left bottom and right top
else if(angle>112.5 && angle<=157.5){
if ((j-1 >= 0 && i-1 >= 0 ) &&(i+1 < height && j+1 < width)) {
if (*(Mag+(i+1)*width+(j-1))>*(Mag+i*width+j) || *(Mag+(i-1)*width+(j+1))>*(Mag+i*width+j)) {
*(*sup+i*width+j)=0;
}
}
}
}
}

}*/
__global__
void supressionGPU(float *sup, float *Mag, float *Dir, int width, int height){

int i,j;
i=threadIdx.x+blockIdx.x*blockDim.x;
j=threadIdx.y+blockIdx.y*blockDim.y;
sup[i*width+j]=Mag[i*width+j];
if(i<height&&j<width){

float angle = Dir[i*width+j];

if(angle<0) angle = angle + M_PI;

angle=(180/M_PI)*angle;

// top and bottom
if(angle > 157.5 || angle <= 22.5){
if (i-1 >= 0 && i+1 < height) {
if (Mag[(i-1)*width+j]>Mag[i*width+j] || Mag[(i+1)*width+j]>Mag[i*width+j])
sup[i*width+j]=0;

}

}
/*        // top left and right botom
else if (angle>22.5 && angle<=67.5) {
if ( (i-1 >= 0 && j-1 >= 0) || ( i+1 < height && j+1 < width)) {
if (Mag[(i-1)*width+(j-1)]> Mag[i*width+j] || Mag[(i+1)*width+(j+1)]>Mag[i*width+j]) {
sup[i*width+j]=0;
}

}

}
*/

// top left and right botom
else if (angle>22.5 && angle<=67.5) {
if ( (i-1) >= 0 && (j-1) >= 0){
if (Mag[(i-1)*width+(j-1)] > Mag[i*width+j]){
sup[i*width+j]=0;
}else if((i+1<height && j+1 <width)){
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
if ((j-1 >= 0 && i-1 >= 0 ) &&(i+1 < height && j+1 < width)) {
if (Mag[(i+1)*width+(j-1)]>Mag[i*width+j] || Mag[(i-1)*width+(j+1)]>Mag[i*width+j]) {
sup[i*width+j]=0;
}
}
}

}

}
/*
void hysteresis(float *sup, float **hys, int height, int width){
float t_high, t_low;
*hys = (float*)malloc(sizeof(float)*height*width);
float *sort = (float*)malloc(sizeof(float)*height*width);
memcpy(*hys, sup, sizeof(float)*height*width);
memcpy(sort, sup, sizeof(float)*height*width);
qsort(sort, height*width, sizeof(float), compare);

t_high = *(sort+(int)(.95*height*width));
t_low = t_high/5;


for(int i=0; i<height; i++){
for(int j=0; j<width; j++){


if(*(sup+i*width+j)>=t_high)
*(*hys+i*width+j)=255;
else if(*(sup+i*width+j)<=t_low)
*(*hys+i*width+j)=0;
else if(*(sup+i*width+j)<t_high && *(sup+i*width+j)>t_low)
*(*hys+i*width+j)=125;
}
}
free(sort);
}
*/
__global__
void hysteresisGPU(float *sup, float *hys, int height, int width, float t_high, float t_low){

int i,j;
i=threadIdx.x+blockIdx.x*blockDim.x;
j=threadIdx.y+blockIdx.y*blockDim.y;

hys[i*width+j]=sup[i*width+j];

if(i<height && j <width){
if(sup[i*width+j]>=t_high)
hys[i*width+j]=255;
else if(sup[i*width+j]<=t_low)
hys[i*width+j]=0;
else if(sup[i*width+j]<t_high && sup[i*width+j]>t_low)
hys[i*width+j]=125;
}
}

/*
void Finaledge(float **edge, float *hys, int height, int width ){

*edge = (float*)malloc(sizeof(float)*height*width);
memcpy(*edge, hys, sizeof(float)*height*width);

for(int i=0; i<height; i++){
for(int j=0; j<width; j++){
for (int y=-1; y<=1; y++){
for (int x=-1; x<=1; x++){
if(i+y<height && i+y>0 && j+x<width && j+x> 0){
if (*(hys+(i+y)*width+x+j)==255)

*(*edge+i*width+j)=255;

else

*(*edge+i*width+j)=0;
}
}
}
}
}
}

*/
__global__
void FinaledgeGPU(float *edge, float *hys, int height, int width ){
int i,j;
i=threadIdx.x+blockIdx.x*blockDim.x;
j=threadIdx.y+blockIdx.y*blockDim.y;
edge[i*width+j]=hys[i*width+j];


for (int y=-1; y<=1; y++){
for (int x=-1; x<=1; x++){
if(i+y<height && i+y>0 && j+x<width && j+x> 0){
if (hys[(i+y)*width+x+j]==255)

edge[i*width+j]=255;

else

edge[i*width+j]=0;
}
}
}
}




/*
void feature_detec(int height, int width, float *vertical, float *horizon){

float k = 0.04;
int window_width = 7;
float *cornerness = (float*)malloc(sizeof(float)*height*width);
float *C_hor = (float*)malloc(sizeof(float)*height*width);
float *C_ver = (float*)malloc(sizeof(float)*height*width);
float Ixx,Iyy,IxIy;
int locatI,locatJ;
struct location *loc;
//FILE *fi;

memcpy(C_hor, horizon, sizeof(float)*height*width);
memcpy(C_ver, vertical, sizeof(float)*height*width);



for (int i=0; i<height; i++){
for (int j=0; j<width; j++){

Ixx = 0;
Iyy = 0;
IxIy = 0;

for(int k = -window_width/2; k < window_width/2 ; k++){
for(int m = -window_width/2; m < window_width/2 ; m++){
int offseti = i+k;
int offsetj = j+m;
if(offseti >= 0 && offsetj >=0 && offseti < height && offsetj < width){
//cornerness = (ad - bc ) - (a+d)^2
Ixx = Ixx + pow(C_ver[offseti+width+offsetj],2);
Iyy = Iyy + pow(C_hor[offseti+width+offsetj],2);
IxIy = IxIy + C_ver[offseti+width+offsetj] * C_hor[offseti+width+offsetj];
}
}
}
*(cornerness+i*width+j)= Ixx*Iyy - pow(IxIy,2) - 0.04* pow((Ixx+Iyy),2);
}
}

window_width = width/32;
int window_heigh = height/32;


loc = (struct location*)malloc(sizeof(struct location)*window_heigh*window_width);
int locount = 0;

for (int i=0; i<height; i=i+window_heigh){
for(int j=0; j<width; j=j+window_width){


locatI = i;
locatJ = j;

for (int k=0; k<window_heigh; k++){
for(int m=0; m<window_width; m++){
if(j+m >= 0 && i+k >=0 && j+m < width && i+k < height)

if (*(cornerness+(i+k)*width+(j+m))>*(cornerness+(locatI)*width+(locatJ))){

locatI = i+k;
locatJ = j+m;

}
}
}

if (locount< window_width*window_heigh){
loc[locount].locationI = locatI;
loc[locount].locationJ = locatJ;
locount+=1;
}
}
}

free(C_hor);
free(C_ver);
}

*/

__global__
void feature_detecGPU(float *d_cornerness,int height, int width, float *C_ver, float *C_hor, int blocksize){

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
// int locatI,locatJ;
// struct location *loc;
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
//    if(globIx < height && globIy <width){

for(int k = -window_width/2; k < window_width/2 ; k++){
for(int m = -window_width/2; m < window_width/2 ; m++){

if(locaIx+k >= 0 && locaIx+k < blockDim.x && locaIy+m >= 0 && locaIy+m < blockDim.y){
int offseti = locaIx+k;
int offsetj = locaIy+m;

Ixx = Ixx + pow(Vshared[offseti*blockDim.y+offsetj],2);
Iyy = Iyy + pow(Hshared[offseti*blockDim.y+offsetj],2);
IxIy = IxIy + Vshared[offseti*blockDim.y+offsetj] * Hshared[offseti*blockDim.y+offsetj];
}

else if(globIx+k >= 0 && globIx+k < height && globIy+m >= 0 && globIy+m < width){
int offseti = globIx+k;
int offsetj = globIy+m;

Ixx = Ixx + pow(C_ver[offseti*width+offsetj],2);
Iyy = Iyy + pow(C_hor[offseti*width+offsetj],2);
IxIy = IxIy + C_ver[offseti*width+offsetj] * C_hor[offseti*width+offsetj];
}
}
}  __syncthreads();
d_cornerness[globIx*width+globIy]= (Ixx*Iyy) - (IxIy*IxIy) - 0.04*(Ixx+Iyy)*(Ixx+Iyy);


}

/*
//window_width = width/32;
//int window_heigh = height/32;
window_width = blockDim.y;
//int window_heigh = height/32;


loc = (struct location*)malloc(sizeof(struct location)*window_heigh*window_width);
int locount = 0;

for (int i=0; i<height; i=i+window_heigh){
for(int j=0; j<width; j=j+window_width){


locatI = i;
locatJ = j;

for (int k=0; k<window_heigh; k++){
for(int m=0; m<window_width; m++){
if(j+m >= 0 && i+k >=0 && j+m < width && i+k < height)

if (*(cornerness+(i+k)*width+(j+m))>*(cornerness+(locatI)*width+(locatJ))){

locatI = i+k;
locatJ = j+m;

}
}
}

if (locount< window_width*window_heigh){
loc[locount].locationI = locatI;
loc[locount].locationJ = locatJ;
locount+=1;
}
}
}

*/

__global__
void find_featureGPU(float *output, float *d_cornerness,int height, int width, int blocksize){

int locatI,locatJ,stride,localIndex;
//    struct location *loc;
int window_size = blockDim.x;
int window_width = blockDim.y;
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
for ( stride = ((window_size*window_size)/2);stride >= 1; stride/=2){
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


//    printf("last decide: %d\n",indexShared[0]);
}

if(locaIx == 0 && locaIy == 0){
output[globIx*width+globIy]=indexShared[0];
int a = indexShared[0];
printf("index;%d ",a);

}



}


/*

for (int i=0; i<height; i=i+window_heigh){
for(int j=0; j<width; j=j+window_width){


locatI = i;
locatJ = j;

for (int k=0; k<window_heigh; k++){
for(int m=0; m<window_width; m++){
if(j+m >= 0 && i+k >=0 && j+m < width && i+k < height)

if (*(cornerness+(i+k)*width+(j+m))>*(cornerness+(locatI)*width+(locatJ))){

locatI = i+k;
locatJ = j+m;

}
}
}

if (locount< window_width*window_heigh){
loc[locount].locationI = locatI;
loc[locount].locationJ = locatJ;
locount+=1;
}
}
}

}

*/
int main(int argc, char ** argv){


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
int blocksize = atof(argv[3]);
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
/*    float t_high, t_low;
float *sort = (float*)malloc(sizeof(float)*height*width);
memcpy(sort, sup, sizeof(float)*height*width);
qsort(sort, height*width, sizeof(float), compare);
t_high = *(sort+(int)(.9*height*width));
t_low = t_high/5;
free(sort); */
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

//cudaMemcpy(features,d_features,sizeof(float)*width*height,cudaMemcpyDeviceToHost);
gettimeofday(&computationend, NULL);

//    FILE  *T0;
//    T0=fopen("index.txt","w");
//fwrite(features,1,height*width,T0);
//fclose(T0);


//printf("%d\n",cornerness);

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
write_image_template("cornerness.pgm", cornerness, width, height);

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

}




               
