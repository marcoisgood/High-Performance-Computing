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

        gettimeofday(&start, NULL);

        read_image_template(argv[1],&image,&width,&height);
        
       	gettimeofday(&computationstart, NULL);

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
    	chunk_image_ghost = ghost_chunk(chunk_image, height/comm_size, width, (k_w/2) ,comm_size, comm_rank);
 

 	temp_horizon = (float*)malloc(sizeof(float)*width*(height/comm_size));
    
	convolve(chunk_image_ghost, temp_horizon, gauss_Kernel, height/comm_size, width, k_w, 1, comm_size, comm_rank,blocksize);
    


	chunk_image_ghost2 = ghost_chunk(temp_horizon, height/comm_size, width, floor(k_w/2),comm_size, comm_rank);    
    	h_temp_chunk = (float*)malloc(sizeof(float)*width*(height/comm_size));
 	convolve(chunk_image_ghost2, h_temp_chunk, dgauss_Kernel, height/comm_size, width, 1, k_w, comm_size, comm_rank,blocksize);
//	MPI_Gather(h_temp_chunk, width*(height/comm_size), MPI_FLOAT, horizon,width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Vertical gradient
    

	temp_vertical = (float*)malloc(sizeof(float)*width*(height/comm_size));
    	convolve(chunk_image_ghost, temp_vertical, gauss_Kernel, height/comm_size, width,1, k_w, comm_size, comm_rank,blocksize);
    	
    	chunk_image_ghost2V = ghost_chunk(temp_vertical, height/comm_size, width, floor(k_w/2),comm_size, comm_rank);

	v_temp_chunk = (float*)malloc(sizeof(float)*width*(height/comm_size));
    	convolve(chunk_image_ghost2V, v_temp_chunk, dgauss_Kernel, height/comm_size, width, k_w, 1, comm_size, comm_rank,blocksize);
 //   	MPI_Gather(v_temp_chunk, width*(height/comm_size), MPI_FLOAT, vertical,width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    
    // Magnitude
    	
	temp_Mag = (float*)malloc(sizeof(float)*width*(height/comm_size));
    	
   	Magnitude(v_temp_chunk, h_temp_chunk, temp_Mag, height/comm_size, width,blocksize);
   // 	MPI_Gather(temp_Mag, width*(height/comm_size),MPI_FLOAT, Mag, width*(height/comm_size),MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // Direction
   	temp_Dir = (float*)malloc(sizeof(float)*width*(height/comm_size));
	Direction(v_temp_chunk, h_temp_chunk, temp_Dir, height/comm_size, width,blocksize);
  //  MPI_Gather(temp_Dir, width*(height/comm_size),MPI_FLOAT, Dir, width*(height/comm_size),MPI_FLOAT, 0, MPI_COMM_WORLD);

    // supression
    	chunk_image_ghost_Mag = ghost_chunk(temp_Mag, height/comm_size, width, 1 ,comm_size, comm_rank);
    
	temp_sup = (float*)malloc(sizeof(float)*width*(height/comm_size));
	
	supression (temp_sup, chunk_image_ghost_Mag, temp_Dir, width, height/comm_size, comm_rank, comm_size,blocksize,1);
   MPI_Gather(temp_sup, width*(height/comm_size),MPI_FLOAT, sup, width*(height/comm_size),MPI_FLOAT, 0, MPI_COMM_WORLD);
    
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
/*
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
        
	write_image_template("edge.pgm", edge, width, height);
      	
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
     	free(horizon);
        free(vertical);
        free(Mag);
        free(Dir); 
        free(sup);
        free(hys);
        free(edge);   
        gettimeofday(&end, NULL);
 
      
        //print
        printf("Number of Threads: %d ,",1);
        printf("processors: %d ,", comm_size);
        printf("Image-Height: %d ,",height);
        printf("Image-Width: %d ,",width);
        printf("Sigma: %f ,",atof(argv[2]));
        printf("End to End time: %ld ,",(end.tv_sec *1000000 + end.tv_usec)-(start.tv_sec * 1000000 + start.tv_usec));
        printf("computation time: %ld ,",(computationend.tv_sec *1000000 + computationend.tv_usec)-(computationstart.tv_sec * 1000000 + computationstart.tv_usec));
        printf("\n");


}
        MPI_Finalize();

}    
