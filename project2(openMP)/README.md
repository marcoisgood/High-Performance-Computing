#Analysis of the Execution Time Variation of OpenMp with Canny Edge Detection and Feature Detector
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
The purpose of this report is to analyze the speedup and parallel efficiency by write a serial and parallel code. In Canny Edge Detector and feature detector, we have to implement convolution, Suppressed, and Hysteresis. In those function, the program needs to get and calculate the value of each pixels. This process take up most all of execute time thus by using to parallelize it, we can effectively to observe the the changing of execution time. In addition, running our program on cluster which provide multiple processors allows us to observe the changing of execution time between different number of threads. (From two threads to thirty-two threads)


<h3 id='2'> Implementation </h3>
<h4 id='3'>Canny edge detector</h4>
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
By setting two time zones: end-to-end time and computation time to measure the execution time in serial code and parallel code. we parallelized the for loop most because for loops take up most of all execution time. The environment we used to run the program is a cluster which allows us to use multiple processors. The sizes of images we use were: 12800, 10240, 7680, 4096, 2048, to 1024. And the value of sigmas were 1.25, 1.1, 0.6. Moreover, the threads we test were 2, 4, 8, 16, 32.

<h3 id='5'>Result and analysis</h3>
We have the figures from 8 to 13 to show our Speed-up time and Efficiency results in each sigma.

<img src = "https://i.imgur.com/gHk4r2b.png" width= "300" />
</br>
<img src = "https://i.imgur.com/x9rhk3I.png" width= "300" />
</br>
<img src = "https://i.imgur.com/tBugo69.png" width= "300" />
</br>
Figure10.

<img src = "https://i.imgur.com/eCUeOEm.png" width= "300" />
</br>
<img src = "https://i.imgur.com/BPWoxND.png" width= "300" />
</br>

The execution time are affected by sigma value, size of image and number of threads. According to the execution time we got, we show the Speed-up time and Efficiency. The average of speed-up time is around 2.6. At the first time, I put too much parallel command to parallelize the for loop. Then, I found the execution time didn’t decrease. Instead, it was increasing. I only parallelized the convolve function at the final. So, it appears that more parallelization is not along with less execution time.


<h3 id='6'> Conclusion </h3>
Along with this project, we could understand the execution time will affected by not only the condition we set but also the condition of cluster. My serial code on the first time running, it showed me the execution time of serial code was faster than parallel code. Then after I have run more times, the serial code was slower than first time. According to our results, the open MP can speed-up the program 2.6 times.



