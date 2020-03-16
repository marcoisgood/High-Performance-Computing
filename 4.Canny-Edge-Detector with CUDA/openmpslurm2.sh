#!/bin/bash
#SBATCH --partition=compute   ### Partition
#SBATCH --job-name=Project04 ### Job Name
#SBATCH --time=07:10:00     ### WallTime
#SBATCH --nodes=1           ### Number of Nodes
#SBATCH --ntasks-per-node=1 ### Number of tasks (MPI processes)
##SBATCH --cpus-per-task=1  ### Number of threads per task (OMP threads). Get num threads using program arguments

srun ./program4 Lenna_org_1024.pgm 0.6 8 >>project4csv.csv

srun ./program4 Lenna_org_2048.pgm 0.6 8 >>project4csv.csv

srun ./program4 Lenna_org_4096.pgm 0.6 8 >>project4csv.csv

srun ./program4 Lenna_org_10240.pgm 0.6 8 >>project4csv.csv

srun ./program4 Lenna_org_12800.pgm 0.6 8 >>project4csv.csv


srun ./program4 Lenna_org_1024.pgm 0.6 16 >>project4csv.csv

srun ./program4 Lenna_org_2048.pgm 0.6 16 >>project4csv.csv

srun ./program4 Lenna_org_4096.pgm 0.6 16 >>project4csv.csv

srun ./program4 Lenna_org_10240.pgm 0.6 16 >>project4csv.csv

srun ./program4 Lenna_org_12800.pgm 0.6 16 >>project4csv.csv


srun ./program4 Lenna_org_1024.pgm 0.6 32 >>project4csv.csv

srun ./program4 Lenna_org_2048.pgm 0.6 32 >>project4csv.csv

srun ./program4 Lenna_org_4096.pgm 0.6 32 >>project4csv.csv

srun ./program4 Lenna_org_10240.pgm 0.6 32 >>project4csv.csv

srun ./program4 Lenna_org_12800.pgm 0.6 32 >>project4csv.csv



srun ./program4 Lenna_org_1024.pgm 1.1 8 >>project4csv.csv

srun ./program4 Lenna_org_2048.pgm 1.1 8 >>project4csv.csv

srun ./program4 Lenna_org_4096.pgm 1.1 8 >>project4csv.csv

srun ./program4 Lenna_org_10240.pgm 1.1 8 >>project4csv.csv

srun ./program4 Lenna_org_12800.pgm 1.1 8 >>project4csv.csv


srun ./program4 Lenna_org_1024.pgm 1.1 16 >>project4csv.csv

srun ./program4 Lenna_org_2048.pgm 1.1 16 >>project4csv.csv

srun ./program4 Lenna_org_4096.pgm 1.1 16 >>project4csv.csv

srun ./program4 Lenna_org_10240.pgm 1.1 16 >>project4csv.csv

srun ./program4 Lenna_org_12800.pgm 1.1 16 >>project4csv.csv


srun ./program4 Lenna_org_1024.pgm 1.1 32 >>project4csv.csv

srun ./program4 Lenna_org_2048.pgm 1.1 32 >>project4csv.csv

srun ./program4 Lenna_org_4096.pgm 1.1 32 >>project4csv.csv

srun ./program4 Lenna_org_10240.pgm 1.1 32 >>project4csv.csv

srun ./program4 Lenna_org_12800.pgm 1.1 32 >>project4csv.csv
