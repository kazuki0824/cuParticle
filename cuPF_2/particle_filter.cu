/*
 * particle_filter.cu
 *
 *  Created on: 2018/10/11
 *      Author: maleicacid
 */

#include "devices/devices.cuh"

#include <curand.h>
#include <curand_kernel.h>

#include "modules/emicp.h"
#include "modules/cub_wrapper.cuh"
#include "particle_filter.h"
#include "user/likelihood.cuh"
#include "user/behavior.h"


#include <stdio.h>
static int b_search(float ary[], float key, int imin, int imax) {
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
static inline int sampling(float random_seed, float ary[], int count)
{
	int imax = count - 1;
	int index = b_search(ary, random_seed, 0, imax);
	if (index>=imax) return imax;
	else return index + 1;
}

// 位置情報
extern float3 state;

// パーティクル
float * p;
float2 * dparticle;
float2 hparticle[sample_count];
float3 pf_out_pose;
extern float3 diff;

float * dLikelihood_table;
float hLikelihood_table[sample_count];

static void prepare_particle_likelihood(float3 xy)
{
	// パーティクルの集合について、尤度と位置を別々に確保している
	cudaMalloc((float2**)&dparticle,sample_count * sizeof(float2));
	cudaMalloc((float2**)&dLikelihood_table,sample_count * sizeof(float));
	//TODO: hparticleに撒く・hLikelihood_tableに尤度をセット
	
	cudaMemcpy(dparticle, hparticle, sizeof(float2) * sample_count,cudaMemcpyDeviceToHost);
	cudaMemcpy(dLikelihood_table, hLikelihood_table, sizeof(float2) * sample_count,cudaMemcpyDeviceToHost);
}


curandGenerator_t g;
cudaStream_t stream_1;
cudaStream_t stream_2;
void Init(float x, float y)
{
	state = make_float3(x,y,0);

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

__global__ static void kStep(float2 * particle_device, float2 * LRF_device, float * LT_device,float3 x_y, unsigned int seed, float * map_device, int n_Beam,
							int width, int height, float resolution, float2 center, float3 diff)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	curandState_t s;
	curand_init(seed, idx, 0, &s);

	particle_device[idx] = prediction(particle_device[idx], diff, 0.8 * sqrt(diff.x * diff.x + diff.y * diff.y), &s);
	LT_device[idx] = likelihood(LT_device[idx], particle_device[idx], LRF_device, n_Beam, 
							    map_device, width, height, resolution, center);
}

void Step()
{
	//Prediction update, likelihood(null stream)
	float2* dLRF; cudaHostGetDevicePointer(&dLRF, hLRF, 0);

	kStep<<<64,128>>>(dparticle,dLRF,dLikelihood_table,state, clock(), d_map, nBeam, width, height, resolution, center, diff);

	//Inclusive scan using CUB(null stream)
	float hPrefix[sample_count];
	incl_scan_CDF(dLikelihood_table, hPrefix, sample_count, 0);

	//TODO: ここか、main()の中でICPを行う？タイミングは任せる
	float2 hICP_result;

	//Wait
	cudaStreamSynchronize(stream_1);
	//Copy ICP result(null stream/blocking)

	//Generate random values for "next" re-sampling
	float* dp; cudaHostGetDevicePointer(&dp, p, 0);
	curandGenerateUniform(g, dp, sample_count);

	//Resampling (Binary search method)
	float max_l = hPrefix[sample_count-1];
	float2 new_particles[sample_count] = {0};
	float new_likelihood[sample_count] = {0};
	for (int var = 0; var < sample_count; ++var) {
		int particle_index = sampling(p[var] * max_l, hPrefix, sample_count);
		new_particles[var] = hparticle[particle_index];
		new_likelihood[var] = hLikelihood_table[particle_index];
	}
	// Update hLikelihood_table
	float likelihood_max = 0;
	for (int i = 0; i < sample_count; ++i)
	{
		hLikelihood_table[i] = new_likelihood[i];
		// 最も尤度の高いパーティクルを現在の姿勢として採用する
		if(hLikelihood_table[i] > likelihood_max)
		{
			likelihood_max = hLikelihood_table[i];
			pf_out_pose.x = new_particles[i].x;
			pf_out_pose.y = new_particles[i].y;
		}
	}

	//Send re-sampled particles to GPU
	cudaMemcpy(hparticle, new_particles, sizeof(float2) * sample_count,cudaMemcpyHostToHost);
	cudaMemcpy(dparticle, hparticle, sizeof(float2) * sample_count,cudaMemcpyDeviceToHost);
	cudaMemcpy(dLikelihood_table, hLikelihood_table, sizeof(float2) * sample_count,cudaMemcpyDeviceToHost);

}
