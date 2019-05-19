extern "C" {
void convolve(float *image, float *output, float *kernel, int height,
              int width, int k_height, int k_width, int comm_size, int comm_rank, int blocksize);
void Magnitude(float *vertical, float *horizon,float *Mag, int height, int width, int blocksize);
void Direction(float *vertical, float *horizon, float *Dir, int height,int width, int blocksize);
void supression(float *sup, float *Mag, float *Dir, int width, int height, int comm_rank, int comm_size, int blocksize, int a);
void hysteresis(float *sup, float *hys, int height, int width, float t_high, float t_low, int blocksize);
void Finaledge(float *edge, float *hys, int height, int width, int comm_size, int comm_rank, int blocksize);
void feature_detec(float *feature, int height, int width, float *vertical, float *horizon, int comm_size, int comm_rank, int blocksize, int a);
void find_feature(float *output,float *temp_feature,int height,int width,int comm_size, int comm_rank,int blocksize);



}
