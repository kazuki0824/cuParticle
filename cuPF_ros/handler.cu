/*
 * handler.cu
 *
 *  Created on: 2018/04/22
 *      Author: kazuki
 */


#include "kernel.hcu"

__global__ void fillparticle(){

}
//すれっどは512まで
	//ブロックは65535*65535
__host__ void prep(int n)
{

	unsigned int seed,id;
	curandState_t s;
	curand_init(seed, id, 0, &s);

	int blocks, threads;
	getNumBlocksAndThreads(5,n,16,512,blocks, threads);

	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);


	float3 current;







}
