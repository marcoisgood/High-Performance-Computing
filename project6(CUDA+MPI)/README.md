#Accelerating Canny edge detector using CUDA+MPI 
---
**01/2019 ECPE 251 High-Performance Computing  
Professor: Venkittaraman Krishnamani </br> University of the Pacific**
___

#### Catalog
- [Introduction](#1)
- [Implementation](#2)
	- [Canny edge detector with CUDA](#3)
	- [CUDA+MPI](#4)
	- [CUDA Aware MPI](#5)
	- [Remote Memory Access](#6)
	- [Unified Virtual Addressing (UVA)](#7)
- [Result and analysis](#8)
- [Conclusion](#9)

---

<h3 id='1'>Introduce</h3>
Our motivation is due to image processing algorithms need more computational power for higher image size like 102400x102400 size image. High-performance computing (HPC) architectures are becoming increasingly more heterogeneous in a move towards exascale. In this project, we are accelerating the Canny edge detector algorithm by combined the CUDA and MPI both. And we will see how's the efficiency of speed-up. 


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

<h4 id='4'>CUDA+MPI</h4>

* CUDA and MPI can be considered separate entities
	*  CUDA handles parallelization on GPU
	* MPI handles parallelization over nodes
* Use one MPI process per GPU and accelerate the computational kernels with CUDA 
* To transfer data between to devices 
	* Sender: Copy data from device to temporary   host buffer 
	* Sender: Send host buffer data 
	* Receiver: Receive data to host buffer 
	* Receiver: Copy data to device
<img src="https://i.imgur.com/tzvDWsY.png" />

*	MPI implementation only pointers to host memory can be passed to MPI

* MPI+GPU: need to send GPU buffers instead of host buffers

* GPU buffers through host memory using cudaMemcpy

> cudaMemcpy(output,d_temp_horizon,sizeof(float)*width*height,cudaMemcpyDeviceToHost);


<h4 id='5'> CUDA Aware MPI </h4>

* CUDA-Aware MPI is communicating data between difference GPU processors without staging through system memory.
* GPU buffers via MPI w/o explicit calls to cudaMemcpy
* Inter- and intra-node GPU communication:
	* GPUDirect Peer-to-Peer (P2P)
	* GPUDirect RDMA (GDR)
* No modifications required to MPI calls
* Supported libraries: OpenMPI, MVAPICH, etc.

<h4 id='6'> Remote Memory Access </h4>
* CUDA 5.0 supports for RDMA- buffers can be directly sent from the GPU memory to a network adapter without staging through host memory.

* Peer-to-Peer (P2P) buffers can be directly copied between the memories of two GPUs

<img src ="https://i.imgur.com/Ed7RiWR.png" />
<img src ="https://i.imgur.com/6IdqFhL.png" />


<h4 id='7'> Unified Virtual Addressing (UVA) </h4>
* One address space for all CPU and GPU memory.
* Determine physical memory location from a pointer value 
* Enable libraries to simplify their interfaces (e.g. MPI and cudaMemcpy)

<img src ="https://i.imgur.com/mwCr1Xq.png" />

<h4 id='8'>Result and analysis</h4>
* Performance Metric: parallel computation time, serial computation 	time, speedup 

<img src ="https://i.imgur.com/r0NlYMD.png" width =500/>
<img src ="https://i.imgur.com/5S9v5Jm.png" width =500/>
<img src ="https://i.imgur.com/W3AxWHW.png" width =500/>

<h3 id='9'> Conclusion </h3>
* Indirect usage of GPU buffers affects computation time
* CUDA+MPI speedup is not better than CUDA 
* Probably CUDA aware MPI could overcome those problems

