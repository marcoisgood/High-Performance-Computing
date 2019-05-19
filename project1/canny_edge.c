#include<stdio.h>
#include "stdlib.h"
#include<math.h>
#include"image_template.h"


    int compare(const void *a, const void *b){

                float c = *(float*)a;
                float d = *(float*)b;
                    if(c < d) return -1;  
                        else return 1;
}

     void convolve(float *image, float **output, float *kernel, int height, 
		int width, int k_height, int k_width){
	
	    *output = (float*)malloc(sizeof(float)*height*width);
	
	    for(int i=0; i<height; i++){
		    for(int j=0; j<width; j++){
			    float sum = 0;
			    for(int k=0; k<k_height;k++){
				    for(int m=0; m<k_width; m++){
                        int offseti = -1*floor(k_height/2)+k;
					    int offsetj = -1*floor(k_width/2)+m;
                            if (j+offsetj >= 0 && (j+offsetj)< height && i+offseti >=0 && i+offseti < width) {
						        sum+= *(image+(i+offseti)*width+j+offsetj)*(*(kernel+(k*k_width)+m));
                            
                                }
                            }
                        }
                            *(*output+(i*width)+j)=sum;

		            }
	            }
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
                 
                   }}
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
 


int main(int argc, char ** argv){
    
        int height,width,k_w;
        float *gauss_Kernel,*Mag, *sup,
        *Dir,*image, *temp_horizon, *horizon,
        *vertical, *temp_vertical, 
        *dgauss_Kernel, *hys, *edge, *Feat;
        
    
        read_image_template(argv[1],&image,&width,&height);
        

        Cal_gauss_kernel(&gauss_Kernel,atof(argv[2]), &k_w, &dgauss_Kernel);
        printf("Gaussian Kernel:%d\n",k_w);
        
		

        convolve(image, &temp_horizon, gauss_Kernel, height, width, k_w, 1 );
        convolve(temp_horizon, &horizon, dgauss_Kernel, height, width, 1, k_w);


        
        convolve(image, &temp_vertical, gauss_Kernel, height, width,1, k_w);
        convolve(temp_vertical, &vertical, dgauss_Kernel, height, width, k_w, 1);


        Magnitude(vertical, horizon, &Mag, height, width);
        Direction(vertical, horizon, &Dir, height, width);
        
        supression (&sup, Mag, Dir, height, width);
        hysteresis (sup, &hys, height, width);
        Finaledge(&edge, hys, height, width);




        //write
        write_image_template("h_convolve.pgm",horizon, width, height);
        write_image_template("v_convolve.pgm",vertical, width, height);
        write_image_template("Magnitude.pgm",Mag, width, height);
        write_image_template("Direction.pgm",Dir, width, height);
        write_image_template("suppress.pgm", sup, width, height);
        write_image_template("suppress.pgm", hys, width, height);
        write_image_template("edge.pgm", edge, width, height);


    }



