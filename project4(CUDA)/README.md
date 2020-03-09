#Analysis of the Execution Time Variation of CUDA with Canny Edge Detection
---
**01/2019 ECPE 251 High-Performance Computing  
Professor: Venkittaraman Krishnamani </br> University of the Pacific**
___

#### Catalog
- [Introduction](#1)
- [Implementation](#2)
	- [Canny edge detector](#3)
	- [Parallelization methodology](#4)
- [Result and analysis](#5)
- [Conclusion](#6)

---

<h3 id='1'>Introduce</h3>
The purpose of this report is to analyze the accelerating by designing a serial code and then using CUDA method to speedup. In Canny Edge Detector, there are several kernels, such as convolution, Suppressed, and Hysteresis. In those kernels, the program needs to get and calculate the value of each pixels. They take up execute time the most, thus by executing on GPUs device, we can effectively to observe the reducing of execution time.



<h3 id='2'> Implementation </h3>
<h4 id='3'>Canny edge detector with CUDA</h4>
We used this function to detect the edge of image. In order to effectively reduce our execution time. We used Open MPI and OpenMP to implement our parallelization. In Open MPI process, we need to consider the right communication between each processor for sending and receiving the data.
In the Figures 1 to 15, it appears that the correlation between efficiency and processors and threads. When we used 4 processors to performance it, the efficiency up to 160 percent. The range of efficiency is from 90 (2 processors) to 300 (32 processors). And, Range of speed up is from 1.5 to 9.

We used this function to detect the edge of image. The outputs will show below.

<img src = "https://www.researchgate.net/profile/Hugo_Hidalgo-Silva/publication/253682881/figure/fig2/AS:298135596355602@1448092473989/Lena-Original-Image-512x512-pixels.png" width= "300"/>
</br>
Figure1. Original

<img src = "https://i.imgur.com/jyEFMza.png" width= "300" />
</br>
Figure2. Horizontal convolve

<img src = "https://i.imgur.com/DpQFF4h.png" width= "300" />
</br>
Figure3. Vertical convolve

<img src = "https://i.imgur.com/JG6L8dL.png" width= "300" />
</br>
Figure4. Direction

<img src = "https://i.imgur.com/It1w4z3.png" width= "300" />
</br>
Figure5. Magnitude

<img src = "https://i.imgur.com/BAiT02k.png" width= "300" />
</br>
Figure6. Hysteresis

<img src = "https://i.imgur.com/XoQKYuZ.png" width= "300" />
</br>
Figure7. Suppress

<img src = "https://i.imgur.com/kOs0FFR.png" width= "300" />
</br>
Figure8. Result


<h4 id='4'>Parallelization methodology</h4>
By setting two time zones: end-to-end time and computation time to measure the execution time in serial code and CUDA code. By setting different numbers of threads per block: 8x8, 16x16, and 32x32, we can observe the difference of speed up. We also set two sigmas: 0.6 and 1.1, 5 different image sizes: 1024x1024, 2048x2048, 4096x4096, 10240x10240, and 12800x12800, and two kernels for convolution: shared memory and global memory.


<h3 id='5'>Result and analysis</h3>
We have the figures from 1 to 6 to show our Speed-up time and results in each sigma, number of thread per block, and different kernel with different size of images. The unit of time is Microsecond(MS)


<img src = "https://i.imgur.com/ezRnL2k.png" width= "1000" />
</br>
<img src = "https://i.imgur.com/ZvO86Bq.png" width= "1000" />
</br>
<img src = "https://i.imgur.com/V1h5t3S.png" width= "1000" />
</br>

<img src = "https://i.imgur.com/n4YvveM.png" width= "1000" />
</br>
<img src = "https://i.imgur.com/PvdAWlD.png" width= "1000" />
</br>

The execution time are affected by sigma value, size of image, number of threads per block, and different kernel using. According to the execution time we got, it shows that using shared memory is faster than only using global memory. By the different numbers of thread per block, we can calculate occupancy of each multiprocessor rate. The occupancy rates in here appear that the rates are justified.


<h3 id='6'> Conclusion </h3>
In this project, we could learn how to use different numbers of thread, block and shared memory to optimize our execution efficiency. The most import thing we need to be careful is the bound of access. In my project, the suppression function was the reason causing the segmentation fault. But this problem only showed up when I run with 10240x10240 and 12400x12400 size images. It was running successfully with the smaller image. But, by using the memory checking tool , we can easily to find where the problem was.


