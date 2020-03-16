rm run_CUDA.csv
for ((i=0;i<5;i++))
do
for ((k=8; k<=32 ; K+4))
        do
for ((j=0;j<50;j++))
do
srun ./canny_edge_CUDA Lenna_org_256.pgm 0.6 $k >> run_CUDA.csv
done
for ((j=0;j<50;j++))
do
srun ./canny_edge_CUDA Lenna_org_512.pgm 0.6 $k >> run_CUDA.csv
done
for ((j=0;j<50;j++))
do
srun ./canny_edge_CUDA Lenna_org_1024.pgm 0.6 $k >> run_CUDA.csv
done
for ((j=0;j<50;j++))
do
srun ./canny_edge_CUDA Lenna_org_2048.pgm 0.6 $k >> run_CUDA.csv
done
for ((j=0;j<50;j++))
do
srun ./canny_edge_CUDA Lenna_org_4096.pgm 0.6 $k >> run_CUDA.csv
done
for ((j=0;j<50;j++))
do
srun ./canny_edge_CUDA Lenna_org_7680.pgm 0.6 $k >> run_CUDA.csv
done
for ((j=0;j<50;j++))
do
srun ./canny_edge_CUDA Lenna_org_10240.pgm 0.6 $k >> run_CUDA.csv
done

        done

done
