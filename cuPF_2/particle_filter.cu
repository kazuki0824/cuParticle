/*
 * particle_filter.cu
 *
 *  Created on: 2018/10/11
 *      Author: maleicacid
 */

#include <curand.h>
#include <curand_kernel.h>

#include "modules/emicp.cuh"
#include "modules/cub_wrapper.cuh"
#include "particle_filter.h"
#include "user/likelihood.h"
#include "user/behavior.h"

#include <stdio.h>
int b_search(float ary[], float key, int imin, int imax) {
    if (imax < imin) {
        return imax;
    } else {
        int imid = imin + (imax - imin) / 2;
        if (ary[imid] > key) {
            return b_search(ary, key, imin, imid - 1);
        } else if (ary[imid] < key) {
            return b_search(ary, key, imid + 1, imax);
        } else {
            return imid;
        }
    }
}


/*********************************************************/

__device__ float map[MAP_SIZE*MAP_SIZE];
static void SetupLMap(float * from, size_t count)
{
	cudaMemcpyToSymbol(map, from, count);
}

/*********************************************************/

float2 state = {0};
float * p;
float2 * dparticle;
float * dLikelihood_table;
float2 * hLRF;
static void prepare_particle_likelihood(float2 xy)
{
	cudaMalloc((float2**)&dparticle,8192 *sizeof(float2));
	cudaMalloc((float2**)&dLikelihood_table,8192 *sizeof(float));
	//TODO: 撒く・尤度をセット


}


curandGenerator_t g;
cudaStream_t stream_1;
cudaStream_t stream_2;
void Init(float x, float y)
{
	state = make_float2(x,y);

	//TODO: 尤度マップ転送
	SetupLMap(NULL, MAP_SIZE*MAP_SIZE);

	//Init RNG (Host)
	curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(g,clock());
	cudaHostAlloc(&p, sample_count * sizeof(float), cudaHostAllocMapped);

	//Generate random values for first re-sampling
	float* dp; cudaHostGetDevicePointer(&dp, p, 0);
	curandGenerateUniform(g, dp, sample_count);

	//Initialize particles
	prepare_particle_likelihood(state);

	//LRF zero-copy
	cudaHostAlloc(&hLRF, sample_count * sizeof(float), cudaHostAllocWriteCombined);

	//Create a new stream
	cudaStreamCreate(&stream_1);
	cudaStreamCreate(&stream_2);
}

__global__ static void kStep(float2 * particle_device, float2 * LRF_device, float * LT_device,float2 x_y, unsigned int seed,vParameter param, float * map_device)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	curandState_t s;
	curand_init(seed, idx, 0, &s);

	particle_device[idx] = prediction(particle_device[idx],param,&s);
	LT_device[idx] = likelihood(LT_device[idx], particle_device[idx], LRF_device, map_device);
}
void Step(vParameter param)
{
	//Prediction update, likelihood(null stream)
	float2* dLRF; cudaHostGetDevicePointer(&dLRF, hLRF, 0);
	kStep<<<64,128>>>(dparticle,dLRF,dLikelihood_table,state, clock(), param, map);

	//Inclusive scan using CUB(null stream)
	float hPrefix[8192];
	incl_scan_CDF(dLikelihood_table, hPrefix, sample_count, 0);

	//TODO: ICP matching(non-null stream1)
	float2 hICP_result;



	//Wait
	cudaStreamSynchronize(stream_1);
	//Copy ICP result(null stream/blocking)

	//Generate random values for "next" re-sampling
	float* dp; cudaHostGetDevicePointer(&dp, p, 0);
	curandGenerateUniform(g, dp, sample_count);

	//Resampling (Binary search method)


}
