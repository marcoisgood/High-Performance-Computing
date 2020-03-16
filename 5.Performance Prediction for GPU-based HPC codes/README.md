#Performance Prediction for GPU-based HPC codes
---
**01/2019 ECPE 251 High-Performance Computing  
Professor: Venkittaraman Krishnamani </br> University of the Pacific**
___

#### Catalog
- [Introduction](#1)
- [Implementation](#2)
	- [Canny edge detector with CUDA](#3)
	- [Corner detection on GPU](#4)
	- [Testing](#5)
	- [Performance Modeling](#6)
- [Result and analysis](#7)
- [Conclusion](#8)

---

<h3 id='1'>Introduce</h3>
The purpose of this report is to analyze the accelerating by designing a serial code and then using CUDA method to speedup. In Canny Edge Detector, there are several kernels, such as convolution, Suppressed, and Hysteresis. In those kernels, the program needs to get and calculate the value of each pixels. They take up execute time the most, thus by executing on GPUs device, we can effectively to observe the reducing of execution time. In addition, in order to prediction, we divided our execution time into three parts: kernel execution time, memory copy from host to device and memory copy from device to host. So, first we implemented the features detection. This kernel will detect different numbers of features along with different block sizes and image sizes. We tested program with several image sizes which are 256, 512, 1024, 2048, 4096, 8192, 10240. This time we only use one sigma 0.6. And, the most important part is block size. We tested with block sizes: 8,12,16,20,24,28,32. We have run 50 times to get a solid result. Moreover, for each kernel time, we considered that float point computation times and global memory access times to predict our kernel time.


<h3 id='2'> Implementation </h3>
<h4 id='3'>Canny edge detector with CUDA </h4>
We used this function to detect the edge of image. The outputs will show below.
</br>
<img src = "https://www.researchgate.net/profile/Hugo_Hidalgo-Silva/publication/253682881/figure/fig2/AS:298135596355602@1448092473989/Lena-Original-Image-512x512-pixels.png" width= "300"/>
</br>
Figure1. Original
</br>

<img src = "https://i.imgur.com/jyEFMza.png" width= "300" />
</br>
Figure2. Horizontal convolve
</br>

<img src = "https://i.imgur.com/DpQFF4h.png" width= "300" />
</br>
Figure3. Vertical convolve
</br>

<img src = "https://i.imgur.com/JG6L8dL.png" width= "300" />
</br>
Figure4. Direction
</br>

<img src = "https://i.imgur.com/It1w4z3.png" width= "300" />
</br>
Figure5. Magnitude
</br>

<img src = "https://i.imgur.com/BAiT02k.png" width= "300" />
</br>
Figure6. Hysteresis
</br>

<img src = "https://i.imgur.com/XoQKYuZ.png" width= "300" />
</br>
Figure7. Suppress
</br>

<img src = "https://i.imgur.com/kOs0FFR.png" width= "300" />
</br>
Figure8. Result
</br>
</br>


We used this function to detect the edge of image. In order to effectively reduce our execution time. We used GPUs to implement our accelerating. On GPUs device, we can use global memory and shared memory to speed up our program. In convolution part, especially, we used shared memory. Other kernels used only global memory. We only used shared memory for convolution because during the calculation, some data were out of the bound that the threads need to access global memory to get the data what it needs. We can restore our data to shared memory to enhance their speed. For other part of functions, like Suppressed and Hysteresis, we only use global memory for implementation. This is due to there is not calculation which needs the data from different block.


<h4 id='4'>Corner detection on GPU</h4>
In the Corner features detection kernel, the first step is to evaluate cornerness values for each pixel by computing the Z matrix over the neighborhood window (7x7) which is a similar step to convolution. After computing the cornerness values, we need to implement an algorithm which helps us
to find the location of the maximum cornerness in a window of BLOCKSIZExBLOCKSIZE.
The following is how to implement:
1. Set the thread block dimensions as dimBlock(BLOCKSIZE,BLOCKSIZE)
2. Using shared memory, have threads in a thread block find the location of the maximum cornerness value. Have thread- 0 of the block write the location of the maximum corner to the list of corners.
The following figure is an example of corners output with image-size is 256x256 and Block-Size is 8.
<img src = "https://i.imgur.com/XniDUc2.png" height="700" />

It appears that the indexes and the threads location I and J.

<h4 id='5'> Testing </h4>
After We implemented the code. We tested the program on GPU in several sizes of images and Block-Sizes. Image sizes {256, 512, 1024, 2048, 4096, 8192, and 10240} x {BLOCKSIZES 8, 12, 16, 20, 24 ,28, and 32} x {sigma 0.6}. In addition, in order to get a solid execution time, we executed the program in 30 times for each parameter.
In the figure 2, it shows the average execution time in each parameter.

<img src="https://i.imgur.com/aX6DWqo.png"/>

The above executions will help us to develop performance prediction models for execution time prediction.

<h4 id='6'> Performance Modeling </h4>

Our application's runtime is governed by the time taken by GPU kernels, H2D communications, D2H communications, and any CPU computations. Especially, entire code is on GPUs, Gaussian kernel evaluation which is on CPU is way too small for any meaningful modeling. So, we will not consider CPU computations time. For performance modeling, we have several modeling.


1. Modeling CPU-GPU Communications
In this part, we have to implement a simple benchmark
that performs memory copy of a randomly filled floating- point array from CPU host to GPU device (H2D) and from GPU device to CPU host (D2H). For a given array size, we had performed 30 statistical executions to compute the average H2D time and D2H time for these array size: 1 KB, 2 KB, 4 KB, 8 KB, 16 KB, 32 KB, 64 KB, 128 KB, 256 KB, 512 KB, 1 MB, 2 MB, 4 MB, 8 MB, 16 MB, 32 MB, 64 MB, 128 MB, 256 MB, 512 MB.
And, Calculate the bandwidth as (data size/time), plot H2D bandwidth vs. array-size (KB) and observe the trend. (Dependent variable: bandwidth, independent variable: data size). The figures 3 to 5 are H2D time and the figures 6 to 8 are D2H time. The unit of time is microsecond.

<img src = "https://i.imgur.com/pIrsN8t.png" width= "300" />
</br>

<img src = "https://i.imgur.com/yTW2Mnl.png" width= "500" />
</br>

<img src = "https://i.imgur.com/ZUq3aw4.png" width= "500" />
</br>

<img src = "https://i.imgur.com/XygNX0M.png" width= "300" />
</br>

<img src = "https://i.imgur.com/GcIJequ.png" width= "500" />
</br>

<img src = "https://i.imgur.com/5lpcDtM.png" width= "500" />
</br>

2. Modeling Kernel Times

For each of the Canny edge kernels and the cornerness kernel, we need to manually compute the number of floating- point computations and memory accesses per thread. When we multiply these with the total number of threads for that kernel, we can get total floating-point operations for the kernel (FLOPS_kernel) and total memory accesses (BYTES_kernel). Then, we explored the correlation between these two variables and the kernel runtime to decide a candidate mathematical model. (Independent variable: FLOPS_kernel, BYTES_kernel; Dependent variable: kernel time). For the approximation kernel, because we are testing for different values of BLOCKSIZE, the independent variables are FLOPS_kernel, BYTES_kernel, and BLOCKSIZE.
In the figure 9, it shows the FLOPS_kernel, BYTES_kernel, and kernel time in different image size and BLOCKSIZE. And the time unit is microsecond. 
In figure 10, it shows the coefficients, p-values, and R^2 values and we can make the kernel time formula.

<img src = "https://i.imgur.com/cHyZWaA.png" width= "400" />
</br>

<img src = "https://i.imgur.com/hojHbIZ.png" width= "400" />
</br>

<img src = "https://i.imgur.com/mh46ib0.png" width= "200" />
</br>



<h4 id='7'>Result and analysis</h4>
By our prediction modeling, we predict the execution time as:
Execution Predict= Predicted H2D time + Predicted D2H time + Predicted kernel runtimes.
Predict the execution times for these configurations (all with Gaussian sigma=0.6):
1. Image Size 3072x 3072 with approximation algorithm BLOCKSIZE=8, 16, 32.
2.Image Size 5120x5120 with approximation algorithm BLOCKSIZE=8, 16, 32.
3.Image Size 7680x7680 with approximation algorithm BLOCKSIZE=8, 16, 32.
For the above 9 cases, we have a table called "Predicted H2D, D2H, and kernel times" (figure 11.) that gives the predicted H2D time, D2H time, kernel times, and CPU time. Also, another table called "Actual H2D, D2H, and kernel times for the actual times for H2D, D2H, kernels (figure 12.). These are obtained by executing code with the above configurations. The table (figure 13.) called "prediction errors for the tested configuration" that gives the prediction error for each of the above 9 configurations as:
</br> Error = (predicted execution time - actual execution time)/actual execution time * 100%

<img src = "https://i.imgur.com/siPztqv.png" width= "500" />
</br>
<img src = "https://i.imgur.com/g3MFbpq.png" width= "500" />
</br>
<img src = "https://i.imgur.com/5LWMwoI.png" width= "500" />
</br>

As we seen, H2D and D2H execution time will not change by different BLOCKSIZE because two execution time only execute the memory copy. And CPU time are too small to predict. So, kernel time will be the important part focus on. In the tables, for kernel execution time, we got 12% to 26% prediction error which is not too bad due to the formula of kernel time whose multiple R-squared value is up to 0.8148 and adjusted R-squared also up to 0.783. For H2D and D2H formula, multiple R-squared is only around 0.1 to 0.2. And the way to enhance our accuracy of H2D and D2H, we may need more sample data. In addition, in prediction error table, the range of percentage is from -5 to 40 because the accuracy of prediction is affected by many conditions. Sometimes, the execution time is slower than our expectations due to the computing resources are competed.


<h3 id='8'> Conclusion </h3>
One of the strengths of statistics machine learning method is that allows us to do large scale of prediction which is not able to practically test on any devices due to limitation of computing resource. But on the other hand, this is also one of weakness of statistics machine meaning method. In order to getting an accurate prediction, we need to collect as much as possible data or sample for our prediction. If we are not able to collect enough data, it may affect our accuracy of result. The most challenging thing is calculation of global access in the kernel which also uses shared memory. Because not each thread access global memory, we need to apply software method of counting instead of manual counting that helps us to understand more about how threads work .
