#include<stdio.h>
#include <string.h>
#include "stdlib.h"
#include<math.h>
#include"image_template.h"
#include"time.h"
#include "sys/time.h"
#include<mpi.h>
#include"canny_edge.h"

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

int range(int x, int y, int h, int w, int top, int bottom){
    if(x < 0 || x >= w){
        return 0;
    }
    else if(y < 0-top || y > h+bottom){
        return 0;
    }
    else
        return 1;
}
/*
void convolve(float *image, float **output, float *kernel, int height,
              int width, int k_height, int k_width, int comm_size, int comm_rank){
    
    *output = (float*)malloc(sizeof(float)*height*width);
    
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
    
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            float sum = 0;
            for(int k=0; k<k_height;k++){
                for(int m=0; m<k_width; m++){
                    int offseti = -1*floor(k_height/2)+k;
                    int offsetj = -1*floor(k_width/2)+m;
                    if (range(j+offsetj, i+offseti, height, width, top, bottom)) {
                        sum+= *(image+(i+offseti)*width+j+offsetj)*(*(kernel+(k*k_width)+m));
                        
                    }
                }
            }
            *(*output+(i*width)+j)=sum;
            
        }
    }
}

*/




void Cal_gauss_kernel(float **gauss_Kernel, float sigma, int*w, float **dgauss_Kernel){
    float pw = round(2.5*sigma-0.5);
    float sum = 0;
    
    *w = 2*pw+1;
    
    *gauss_Kernel=(float*)malloc(sizeof(float)*(*w));
    for (int i= 0; i<(*w); i++){
        (*gauss_Kernel)[i] = exp((-1*(i-pw)*(i-pw))/(2*sigma*sigma));
        sum +=(*gauss_Kernel)[i];
    }
    for (int i=0; i<(*w); i++){
        (*gauss_Kernel)[i] = (*gauss_Kernel)[i]/sum;
    }
    
    sum = 0;
    
    *dgauss_Kernel=(float*)malloc(sizeof(float)*(*w));
    
    for(int i=0; i<(*w); i++){
        (*dgauss_Kernel)[i] = -1*(i-pw)*exp((-1*(i-pw)*(i-pw))/(2*sigma*sigma));
        sum-=i*(*dgauss_Kernel)[i];
    }
    
    for(int i=0; i<(*w); i++){
        (*dgauss_Kernel)[i]/=sum;
    }
    float temp;
    int start=0, end= *w-1;
    
    while (start < end){
        temp=(*dgauss_Kernel)[start];
        (*dgauss_Kernel)[start]=(*dgauss_Kernel)[end];
        (*dgauss_Kernel)[end]=temp;
        start++;
        end--;
    }
}
/*
void supression(float **sup, float *Mag, float *Dir, int width, int height, int comm_rank, int comm_size){
    *sup = (float*)malloc(sizeof(float)*height*width);
    memcpy(*sup, Mag, sizeof(float)*height*width);
    
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

    for (int i =0 ; i< height; i++){
        for (int j = 0; j< width; j++){
            
            float angle = *(Dir+i*width+j);
            
            if(angle<0) angle = angle + M_PI;
            
            angle=(180/M_PI)*angle;
            
            // top and bottom
            if(angle > 157.5 || angle <= 22.5){
                if (i-1 >= 0-top && i+1 < height+bottom) {
                    if (*(Mag+(i-1)*width+j)>*(Mag+i*width+j) || *(Mag+(i+1)*width+j)>*(Mag+i*width+j)){
                        *(*sup+i*width+j)=0;
                    }
                }
                
            }
            // top left and right botom
            else if (angle>22.5 && angle<=67.5) {
                if ( (i-1 >= 0-top && j-1 >= 0) && ( i+1 < height+bottom && j+1 < width)) {
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
                if ((j-1 >= 0 && i-1 >= 0-top ) &&(i+1 < height+bottom && j+1 < width)) {
                    if (*(Mag+(i+1)*width+(j-1))>*(Mag+i*width+j) || *(Mag+(i-1)*width+(j+1))>*(Mag+i*width+j)) {
                        *(*sup+i*width+j)=0;
                    }
                }
            }
        }
    }
    
}*/
/*
void hysteresis(float *sup, float **hys, int height, int width, float t_high, float t_low){
    *hys = (float*)malloc(sizeof(float)*height*width);
    memcpy(*hys, sup, sizeof(float)*height*width);
    
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
}



void Finaledge(float **edge, float *hys, int height, int width, int comm_size, int comm_rank){
    
    *edge = (float*)malloc(sizeof(float)*height*width);
    memcpy(*edge, hys, sizeof(float)*height*width);
    
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
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            for (int y=-1; y<=1; y++){
                for (int x=-1; x<=1; x++){
                    if(i+y<height+bottom && i+y>0-top && j+x<width && j+x> 0){
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



void feature_detec(int height, int width, float *vertical, float *horizon, int comm_size, int comm_rank){
    
    int window_width = 7;
    float *cornerness = (float*)malloc(sizeof(float)*height*width);
    //float *C_hor = (float*)malloc(sizeof(float)*height*width);
    //float *C_ver = (float*)malloc(sizeof(float)*height*width);
    float Ixx,Iyy,IxIy;
    int locatI,locatJ;
    struct location *loc;
    //FILE *fi;
    
    //memcpy(C_hor, horizon, sizeof(float)*height*width);
    //memcpy(C_ver, vertical, sizeof(float)*height*width);
    
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

    
    
    for (int i=0; i<height; i++){
        for (int j=0; j<width; j++){
            
            Ixx = 0;
            Iyy = 0;
            IxIy = 0;
            
            for(int k = -window_width/2; k <= window_width/2 ; k++){
                for(int m = -window_width/2; m <= window_width/2 ; m++){
                    int offseti = i+k;
                    int offsetj = j+m;
                    if(offseti >= 0-top && offsetj >=0 && offseti < height+bottom && offsetj < width){
                        //cornerness = (ad - bc ) - (a+d)^2
                        Ixx = Ixx + pow(vertical[offseti+width+offsetj],2);
                        Iyy = Iyy + pow(horizon[offseti+width+offsetj],2);
                        IxIy = IxIy + vertical[offseti+width+offsetj] * horizon[offseti+width+offsetj];
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
            
            if (locount<window_width*window_heigh && locount<1000 ){
                loc[locount].locationI = locatI;
                loc[locount].locationJ = locatJ;
                locount+=1;
            }
        }
    }
    
    //free(C_hor);
    //free(C_ver);
}
*/

/*
float *ghost_chunk(float *image, int height, int width, int a, int comm_size, int comm_rank){
    MPI_Status status;
    float *output;
    if(comm_rank == 0 || comm_rank == (comm_size-1))
        output = (float*)malloc(sizeof(float)*width*(height + a));
    
    else
        output = (float*)malloc(sizeof(float)*width*(height + 2*a));
    
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

*/
int main(int argc, char **argv){
   


 

    int height,width,k_w;
    float *gauss_Kernel,*Mag, *sup,
    *Dir,*image, *temp_horizon, *horizon,
    *vertical, *temp_vertical,
    *dgauss_Kernel, *hys, *edge,*corneress,
    *chunk_image_ghost,
    *chunk_image,
    *chunk_image_ghost2,*chunk_image_ghost2V,
    *h_temp_chunk,*v_temp_chunk,
    *temp_Mag,*temp_Dir,
    *temp_sup,*chunk_image_ghost_Mag,
    *temp_hys,t_high, t_low,
    *chunk_image_ghost_hys,*temp_edge,
    *h_chunk_feature,*v_chunk_feature,*temp_feature, *temp_corneress;
    struct timeval start, end, computationstart,computationend ;
    
    	int comm_size,comm_rank;
	int blocksize = atof(argv[3]);    
    
    	Cal_gauss_kernel(&gauss_Kernel,atof(argv[2]), &k_w, &dgauss_Kernel);
    
	MPI_Init(&argc,&argv);
    
    
    	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    	MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    

    if(comm_rank== 0){

       // gettimeofday(&start, NULL);

        read_image_template(argv[1],&image,&width,&height);
        
       // gettimeofday(&computationstart, NULL);

        horizon = (float*)malloc(sizeof(float)*width*height);
        vertical = (float*)malloc(sizeof(float)*width*height);
        Mag = (float*)malloc(sizeof(float)*width*height);
        Dir = (float*)malloc(sizeof(float)*width*height);
        sup = (float*)malloc(sizeof(float)*width*height);
        hys = (float*)malloc(sizeof(float)*width*height);
        edge = (float*)malloc(sizeof(float)*width*height);
	corneress = (float*)malloc(sizeof(float)*width*height);

    }
    
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    chunk_image = (float*)malloc(sizeof(float)*width*(height/comm_size));
    
    MPI_Scatter(image, (height/comm_size)*width,
                MPI_FLOAT, chunk_image, (height/comm_size)*width,
                MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    
    
    //Horizonal gradient
  	temp_horizon = (float*)malloc(sizeof(float)*width*(height/comm_size));
  

	convolve(chunk_image,temp_horizon,gauss_Kernel,dgauss_Kernel,height/comm_size,width,k_w,1,comm_size,comm_rank,blocksize);

/*  	chunk_image_ghost = ghost_chunk(chunk_image, height/comm_size, width, floor(k_w/2) ,comm_size, comm_rank);
 

 	temp_horizon = (float*)malloc(sizeof(float)*width*(height/comm_size));
    
	convolve(chunk_image_ghost, temp_horizon, gauss_Kernel, height/comm_size, width, k_w, 1, comm_size, comm_rank,blocksize);
    


	chunk_image_ghost2 = ghost_chunk(temp_horizon, height/comm_size, width, floor(k_w/2),comm_size, comm_rank);    
    	h_temp_chunk = (float*)malloc(sizeof(float)*width*(height/comm_size));
 	convolve(chunk_image_ghost2, h_temp_chunk, dgauss_Kernel, height/comm_size, width, 1, k_w, comm_size, comm_rank,blocksize);
	
*/
	MPI_Gather(temp_horizon, width*(height/comm_size), MPI_FLOAT, horizon,width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Vertical gradient
    
/*
	temp_vertical = (float*)malloc(sizeof(float)*width*(height/comm_size));
    	convolve(chunk_image_ghost, temp_vertical, gauss_Kernel, height/comm_size, width,1, k_w, comm_size, comm_rank,blocksize);
    	
    	chunk_image_ghost2V = ghost_chunk(temp_vertical, height/comm_size, width, floor(k_w/2),comm_size, comm_rank);

	v_temp_chunk = (float*)malloc(sizeof(float)*width*(height/comm_size));
    	convolve(chunk_image_ghost2V, v_temp_chunk, dgauss_Kernel, height/comm_size, width, k_w, 1, comm_size, comm_rank,blocksize);
    	MPI_Gather(v_temp_chunk, width*(height/comm_size), MPI_FLOAT, vertical,width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    
    // Magnitude
    	
	temp_Mag = (float*)malloc(sizeof(float)*width*(height/comm_size));
    	
   	Magnitude(v_temp_chunk, h_temp_chunk, temp_Mag, height/comm_size, width,blocksize);
    	MPI_Gather(temp_Mag, width*(height/comm_size),MPI_FLOAT, Mag, width*(height/comm_size),MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // Direction
   	temp_Dir = (float*)malloc(sizeof(float)*width*(height/comm_size));
	Direction(v_temp_chunk, h_temp_chunk, temp_Dir, height/comm_size, width,blocksize);
    MPI_Gather(temp_Dir, width*(height/comm_size),MPI_FLOAT, Dir, width*(height/comm_size),
    MPI_FLOAT, 0, MPI_COMM_WORLD);

    // supression
    	chunk_image_ghost_Mag = ghost_chunk(temp_Mag, height/comm_size, width, 1 ,comm_size, comm_rank);
    
	temp_sup = (float*)malloc(sizeof(float)*width*(height/comm_size));
	
	supression (temp_sup, chunk_image_ghost_Mag, temp_Dir, width, height/comm_size, comm_rank, comm_size,blocksize,1);
    MPI_Gather(temp_sup, width*(height/comm_size),MPI_FLOAT, sup, width*(height/comm_size),
    MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // hysteresis
    
    if(comm_rank==0){
    
        float *sort = (float*)malloc(sizeof(float)*height*width);
        memcpy(sort, sup, sizeof(float)*height*width);
        qsort(sort, height*width, sizeof(float), compare);
        t_high = *(sort+(int)(.9*height*width));
        t_low = t_high/5;
        free(sort);
    }

    MPI_Bcast(&t_high, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&t_low, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   
	temp_hys = (float*)malloc(sizeof(float)*width*(height/comm_size));
	
	hysteresis (temp_sup, temp_hys, height/comm_size, width, t_high, t_low,blocksize);
    MPI_Gather(temp_hys, width*(height/comm_size),MPI_FLOAT, hys, width*(height/comm_size),MPI_FLOAT, 0, MPI_COMM_WORLD);


    
  
    
    // Finaledge
    
    
    	chunk_image_ghost_hys = ghost_chunk(temp_hys, height/comm_size, width, 1 ,comm_size, comm_rank);
   
	temp_edge = (float*)malloc(sizeof(float)*width*(height/comm_size));

	Finaledge(temp_edge, chunk_image_ghost_hys, height/comm_size, width, comm_size, comm_rank, blocksize);
    
    	MPI_Gather(temp_edge, width*(height/comm_size), MPI_FLOAT, edge, width*(height/comm_size),MPI_FLOAT, 0, MPI_COMM_WORLD);

    //feature

    	h_chunk_feature = ghost_chunk(h_temp_chunk, height/comm_size, width, 1 ,comm_size, comm_rank);
    	v_chunk_feature = ghost_chunk(v_temp_chunk, height/comm_size, width, 1 ,comm_size, comm_rank);
	temp_feature = (float*)malloc(sizeof(float)*width*(height/comm_size));
	feature_detec(temp_feature, height/comm_size, width, v_chunk_feature, h_chunk_feature, comm_size, comm_rank, blocksize,1);
   //find feature
   	temp_corneress = (float*)malloc(sizeof(float)*width*(height/comm_size));	
   	find_feature(temp_corneress,temp_feature,height,width,comm_size, comm_rank,blocksize); 
 	 
    	MPI_Gather(temp_feature, width*(height/comm_size), MPI_FLOAT, corneress, width*(height/comm_size),MPI_FLOAT, 0, MPI_COMM_WORLD);
*/
  
    //output image
	if(comm_rank == 0){
        gettimeofday(&computationend, NULL);
        write_image_template("h_convolve.pgm",horizon, width, height);
    //    write_image_template("v_convolve.pgm",vertical, width, height);
   //     write_image_template("Magnitude.pgm",Mag, width, height);
   //     write_image_template("Direction.pgm",Dir, width, height);
   //     write_image_template("suppress.pgm", sup, width, height);
  //      write_image_template("hysteresis.pgm", hys, width, height);
 //       write_image_template("edge.pgm", edge, width, height);
      	
/*	int location_I, location_J;

        for(int i = 0; i<width*height; i++){
        
        if (corneress[i]>0){
        int a = *(corneress+i);
        location_I = a/width;
        location_J = a%width;
      
        printf("Index:%d, I:%d, J:%d\n",a,location_I,location_J);
        
        }
}*/
	 
        //free
     /*	  free(horizon);
        free(vertical);
        free(Mag);
        free(Dir); 
        free(sup);
        free(hys);
        free(edge);  */ 
       // gettimeofday(&end, NULL);
 
      
        //print
        printf("Number of Threads: %d ",1);
        printf("processors: %d ", comm_size);
        printf("Image-Height: %d ",height);
        printf("Image-Width: %d ",width);
        printf("Sigma: %f ",atof(argv[2]));
        printf("End to End time: %ld ",(end.tv_sec *1000000 + end.tv_usec)-(start.tv_sec * 1000000 + start.tv_usec));
        printf("computation time: %ld ",(computationend.tv_sec *1000000 + computationend.tv_usec)-(computationstart.tv_sec * 1000000 + computationstart.tv_usec));
        printf("\n");


}
        MPI_Finalize();

}    
