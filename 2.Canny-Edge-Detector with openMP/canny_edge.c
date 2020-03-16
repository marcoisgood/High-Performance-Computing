#include<stdio.h>
#include <string.h>
#include "stdlib.h"
#include<math.h>
#include"image_template.h"
#include"time.h"
#include "omp.h"
#include "sys/time.h"


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

     void convolve(float *image, float **output, float *kernel, int height, 
		int width, int k_height, int k_width){
	
	    *output = (float*)malloc(sizeof(float)*height*width);
	

#pragma omp parallel for
	    for(int i=0; i<height; i++){
		    for(int j=0; j<width; j++){
			     float sum = 0;
			    for(int k=0; k<k_height;k++){
                    for( int m=0; m<k_width; m++){
                        
                        int offseti = -1*floor(k_height/2)+k;
					    int offsetj = -1*floor(k_width/2)+m;
        
                            if(range(j+offsetj, i+offseti, height, width)){
		sum += *(image+(i+offseti)*width+j+offsetj)*(*(kernel+(k*k_width)+m));
                                }
            
        
                            }
                        }
                            *(*output+(i*width)+j)=sum;
		            }
                }
         
        }






    void Cal_gauss_kernel(float **gauss_Kernel, float sigma, int*w, float **dgauss_Kernel){
        float pw = round(2.5*sigma-0.5);
        int sum = 0;
        
        *w = 2*pw+1;
		
        *gauss_Kernel=(float*)malloc(sizeof(float)*(*w));
        *dgauss_Kernel=(float*)malloc(sizeof(float)*(*w));
        

        
		//printf("\n number of threads here:%d",omp_get_num_threads());
		


            for (int i= 0; i<(*w); i++){
                    (*gauss_Kernel)[i] = exp((-1*(i-pw)*(i-pw))/(2*sigma*sigma));
                        sum +=(*gauss_Kernel)[i];
                    }


            for (int i=0; i<(*w); i++){
                    (*gauss_Kernel)[i] = (*gauss_Kernel)[i]/sum;
                }

    
    sum = 0;


	        for(int i=0; i<(*w); i++){
		        (*dgauss_Kernel)[i] = -1*(i-pw)*exp((-1*(i-pw)*(i-pw))/(2*sigma*sigma));
		        sum-=i*(*dgauss_Kernel)[i];			   
                        }

        	for(int i=0; i<(*w); i++){
		        (*dgauss_Kernel)[i]/=sum;	
                }


            }

    void Magnitude(float *vertical, float *horizon,float **Mag, int height, int width){
               
                *Mag = (float*)malloc(sizeof(float)*height*width);
			

    
               for(int i=0; i<height; i++){
                   for(int j=0; j<width; j++){
                       
                       
                      *(*Mag+i*width+j)=sqrt(pow(*(vertical+i*width+j),2) 
						+ pow(*(horizon+i*width+j),2));
                   }
               }

            }

    void Direction(float *vertical, float *horizon, float **Dir, int height,int width){

               *Dir = (float*)malloc(sizeof(float)*height*width);

			
        for(int i=0; i<height; i++){
                   for(int j=0; j<width; j++){
                   
                   
                   *(*Dir+i*width+j)=atan2(*(vertical+i*width+j),*(horizon+i*width+j));
                   }
                }

        }

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
        }
                
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

            void Finaledge(float **edge, float *hys, int height, int width ){
                    
                    *edge = (float*)malloc(sizeof(float)*height*width);
	                memcpy(*edge, hys, sizeof(float)*height*width);	
	               


                    for(int i=0; i<height; i++){
		                for(int j=0; j<width; j++){
                            for ( int y=-1; y<=1; y++){
                                for ( int x=-1; x<=1; x++){
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
        



    void feature_detec(int height, int width, float *vertical, float *horizon){
                
        
                int window_width = 7;
                float *cornerness = (float*)malloc(sizeof(float)*height*width);
                float *C_hor = (float*)malloc(sizeof(float)*height*width);
                float *C_ver = (float*)malloc(sizeof(float)*height*width);
                float Ixx,Iyy,IxIy;
                int locatI,locatJ;
                struct location *loc;
                //FILE *fi;
                int window_heigh;
                
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
                    window_heigh = height/32;
              
                    loc = (struct location*)malloc(sizeof(struct location)*window_heigh*window_width);    
                    int locount = 0;


                for ( int i=0; i<height; i=i+window_heigh){
                    for( int j=0; j<width; j=j+window_width){


                                locatI = i;
                                locatJ = j;
               
                        for (int k=0; k<window_heigh; k++){
                            for( int m=0; m<window_width; m++){
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

                //printf("locationI: %d locationJ: %d\n",loc[locount].locationI,loc[locount].locationJ);

                free(C_hor);
                free(C_ver);
            } 




int main(int argc, char ** argv){
    	
	
        int height,width,k_w;
        //size_t write_ij;
        //FILE *fi;
        float *gauss_Kernel,*Mag, *sup,
        *Dir,*image, *temp_horizon, *horizon,
        *vertical, *temp_vertical, 
        *dgauss_Kernel, *hys, *edge;
		struct timeval start, end, computationstart,computationend ;
	
	gettimeofday(&start, NULL);
	int processors = atoi(argv[3]);

    
	read_image_template(argv[1],&image,&width,&height);
        
		
	
    omp_set_num_threads(processors);
	
    
    gettimeofday(&computationstart, NULL);

	Cal_gauss_kernel(&gauss_Kernel,atof(argv[2]), &k_w, &dgauss_Kernel);
	
	
        //Horizonal gradient
		  
        convolve(image, &temp_horizon, gauss_Kernel, height, width, k_w, 1 );
       
        convolve(temp_horizon, &horizon, dgauss_Kernel, height, width, 1, k_w);


        // Vertical gradient
        	  
        convolve(image, &temp_vertical, gauss_Kernel, height, width,1, k_w);
       		  
        convolve(temp_vertical, &vertical, dgauss_Kernel, height, width, k_w, 1);


        // Magnitude
		  
        Magnitude(vertical, horizon, &Mag, height, width);

        // Direction
         
        Direction(vertical, horizon, &Dir, height, width);
    
        // supression
        supression (&sup, Mag, Dir, height, width);
    

        // hysteresis

        hysteresis (sup, &hys, height, width);

        // Finaledge
        
        Finaledge(&edge, hys, height, width);
    
        // feature
        feature_detec(height, width, vertical, horizon);
    
        gettimeofday(&computationend, NULL);

        //output image
        write_image_template("h_convolve.pgm",horizon, width, height);
        write_image_template("v_convolve.pgm",vertical, width, height);
        write_image_template("Magnitude.pgm",Mag, width, height);
        write_image_template("Direction.pgm",Dir, width, height);
        write_image_template("suppress.pgm", sup, width, height);
        write_image_template("hysteresis.pgm", hys, width, height);
        write_image_template("edge.pgm", edge, width, height);
    
        //free
        free(Mag);
        free(Dir);
        free(horizon);
        free(vertical);
        free(sup);
        free(hys);
        free(edge);
    
    
        gettimeofday(&end, NULL);
        

       printf("Number of Threads:" "%d" "Image-Height:" "%d" "Width:" "%d" "Sigma:" "%f" "End to End time:" "%ld" "computation time:" "%ld\n",processors,height,width,atof(argv[2]),(end.tv_sec *1000000 + end.tv_usec)-(start.tv_sec * 1000000 + start.tv_usec),(computationend.tv_sec *1000000 + computationend.tv_usec)-(computationstart.tv_sec * 1000000 + computationstart.tv_usec));

    

}





