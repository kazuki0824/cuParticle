/*
 * kernel.cu
 *
 *  Created on: 2018/04/21
 *      Author: kazuki
 */


#include <curand_kernel.h>
#include "nd_noise.hcu"
#include "likelihood.hcu"

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel
// 6, we observe the maximum specified number of blocks, because each thread in
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{

    //get device capability, to avoid block/grid size exceed the upper bound
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    if (whichKernel < 3)
    {
        threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
        blocks = (n + threads - 1) / threads;
    }
    else
    {
        threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
    }

    if ((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
    {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > prop.maxGridSize[0])
    {
        printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
               blocks, prop.maxGridSize[0], threads*2, threads);

        blocks /= 2;
        threads *= 2;
    }

    if (whichKernel == 6)
    {
        blocks = MIN(maxBlocks, blocks);
    }
}


template <class T, unsigned int blockSize>
__global__ void
reduce5(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;

    T mySum = (i < n) ? g_idata[i] : 0;

    if (i + blockSize < n)
        mySum += g_idata[i+blockSize];

    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) &&(tid < 128))
    {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = mySum = mySum + sdata[tid +  64];
    }

    __syncthreads();

    if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            mySum += __shfl_down(mySum, offset);
        }
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void
reduce(int size, int threads, int blocks,
       int whichKernel, T *d_idata, T *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    // choose which of the optimized versions of reduction to launch

    switch (threads)
    {
    case 512:
    	reduce5<T, 512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
    	break;

    case 256:
    	reduce5<T, 256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
    	break;

    case 128:
    	reduce5<T, 128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
    	break;

    case 64:
    	reduce5<T,  64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
    	break;

    case 32:
    	reduce5<T,  32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
    	break;

    case 16:
    	reduce5<T,  16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
    	break;

    case  8:
    	reduce5<T,   8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
    	break;

    case  4:
    	reduce5<T,   4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
    	break;

    case  2:
    	reduce5<T,   2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
    	break;

    case  1:
    	reduce5<T,   1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
    	break;
    }

}

// Instantiate the reduction function for 3 types
template void
reduce<int>(int size, int threads, int blocks,
            int whichKernel, int *d_idata, int *d_odata);

template void
reduce<float>(int size, int threads, int blocks,
              int whichKernel, float *d_idata, float *d_odata);

template void
reduce<double>(int size, int threads, int blocks,
               int whichKernel, double *d_idata, double *d_odata);



__device__ void pf(float3 current)
{




	float3 noise = getSystemNoise(&s);

	float3 prediction = suggest_distribution(current ,noise); //TODO:現在の状態"current"はどこから入手するか

	float l = likelihood(prediction); //予測のもっともらしさを計算する

	/*リダクションで尤度の総和を導出*/



}



