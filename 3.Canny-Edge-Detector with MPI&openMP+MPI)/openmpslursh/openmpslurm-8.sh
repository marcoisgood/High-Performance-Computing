#!/bin/bash
#SBATCH --partition=compute   ### Partition
#SBATCH --job-name=Project38 ### Job Name
#SBATCH --time=20:10:00     ### WallTime
#SBATCH --nodes=2           ### Number of Nodes
#SBATCH --ntasks-per-node=8 ### Number of tasks (MPI processes)



rm runsM8.csv

for ((i=2;i<=32;i=i*2))
do

for((j=0;j<30;j++))
do
srun ./canny_edge_Mix Lenna_org_1024.pgm 1.25 $i >>runsM8.csv
done

for((j=0;j<30;j++))
do
srun ./canny_edge_Mix Lenna_org_2048.pgm 1.25 $i >>runsM8.csv
done

for((j=0;j<30;j++))
do
srun ./canny_edge_Mix Lenna_org_4096.pgm 1.25 $i >>runsM8.csv
done

for((j=0;j<30;j++))
do
srun ./canny_edge_Mix Lenna_org_7680.pgm 1.25 $i >>runsM8.csv
done

for((j=0;j<30;j++))
do
srun ./canny_edge_Mix Lenna_org_10240.pgm 1.25 $i >>runsM8.csv
done

for((j=0;j<30;j++))
do
srun ./canny_edge_Mix Lenna_org_12800.pgm 1.25 $i >>runsM8.csv
done

for((j=0;j<30;j++))
do
srun ./canny_edge_Mix Lenna_org_1024.pgm 1.1 $i >>runsM8.csv
done

for((j=0;j<30;j++))
do
srun ./canny_edge_Mix Lenna_org_2048.pgm 1.1 $i >>runsM8.csv
done

for((j=0;j<30;j++))
do
srun ./canny_edge_Mix Lenna_org_4096.pgm 1.1 $i >>runsM8.csv
done

for((j=0;j<30;j++))
do
srun ./canny_edge_Mix Lenna_org_7680.pgm 1.1 $i >>runsM8.csv
done

for((j=0;j<30;j++))
do
srun ./canny_edge_Mix Lenna_org_10240.pgm 1.1 $i >>runsM8.csv
done

for((j=0;j<30;j++))
do
srun ./canny_edge_Mix Lenna_org_12800.pgm 1.1 $i >>runsM8.csv
done

for((j=0;j<30;j++))
do
srun ./canny_edge_Mix Lenna_org_1024.pgm 0.6 $i >>runsM8.csv
done

for((j=0;j<30;j++))
do
srun ./canny_edge_Mix Lenna_org_2048.pgm 0.6 $i >>runsM8.csv
done

for((j=0;j<30;j++))
do
srun ./canny_edge_Mix Lenna_org_4096.pgm 0.6 $i >>runsM8.csv
done

for((j=0;j<30;j++))
do
srun ./canny_edge_Mix Lenna_org_7680.pgm 0.6 $i >>runsM8.csv
done

for((j=0;j<30;j++))
do
srun ./canny_edge_Mix Lenna_org_10240.pgm 0.6 $i >>runsM8.csv
done

for((j=0;j<30;j++))
do
srun ./canny_edge_Mix Lenna_org_12800.pgm 0.6 $i >>runsM8.csv
done

done
