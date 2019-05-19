rm runNew_CUDA.csv



for ((k=8;k<=32;k=k+4))
do
for ((j=0;j<30;j++))
do

./canny_edge_CUDA Lenna_org_256.pgm 0.6 $k >> runNew_CUDA.csv
done
for ((j=0;j<30;j++))
do

./canny_edge_CUDA Lenna_org_512.pgm 0.6 $k >> runNew_CUDA.csv
done
for ((j=0;j<30;j++))
do

./canny_edge_CUDA Lenna_org_1024.pgm 0.6 $k >> runNew_CUDA.csv
done
for ((j=0;j<30;j++))
do

./canny_edge_CUDA Lenna_org_2048.pgm 0.6 $k >> runNew_CUDA.csv
done
for ((j=0;j<30;j++))
do

./canny_edge_CUDA Lenna_org_4096.pgm 0.6 $k >> runNew_CUDA.csv
done
for ((j=0;j<30;j++))
do

./canny_edge_CUDA Lenna_org_7680.pgm 0.6 $k >> runNew_CUDA.csv
done
        done
