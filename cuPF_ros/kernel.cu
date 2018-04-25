/*
 * kernel.cu
 *
 *  Created on: 2018/04/21
 *      Author: kazuki
 */

#include <curand.h>
#include <curand_kernel.h>
#include "nd_noise.hcu"
#include "likelihood.hcu"
#include "scan_largearray_kernel.h"
#include <time.h>
#include <stdio.h>
#include "helper_math.h"

#include "bridge_header.h"

using namespace std;


float2* lrf_host=NULL;
float2* lrf_device;
__host__ float2* getSensorDataPtrHost(){
	return lrf_host;
}

__device__ float pred(float3 * current, float3 current_speed)
{
	unsigned int seed,id;
	curandState_t s;
	curand_init(seed, id, 0, &s);//高速化を図りたい


	mean_var_param pa; //TODO:
	float3 noise = getSystemNoise(&s, pa);
	float3 prediction = suggest_distribution(*current ,noise, current_speed);

	float l = likelihood(prediction); //予測のもっともらしさを計算する

	*current = prediction;

	return l;

}

__global__ void fill_particle(float3 initialState, float3 * p)
{
	unsigned int seed,id;
	curandState_t s;
	curand_init(seed, id, 0, &s);
	int thId=threadIdx.x+ blockDim.x * blockIdx.x;

	mean_var_param pa ; //TODO:
	p[thId] = initialState + getSystemNoise(&s, pa);
}

__global__ void particle(float3 *particles, float *importance, float3 current_speed, float3 *__state_ptr)
{
	int thId=threadIdx.x+ blockDim.x * blockIdx.x;
	importance[thId] = pred(&particles[thId], current_speed);

	//並列リダクションで最大値を求める



	__state_ptr;
}


float3 * particle_set;
float * importance;
float * importance_prefix;

curandGenerator_t g;
float* p;

static float3* __current_state;

/*リダクションで尤度の総和を導出*/
float3* start(float3 state) {
	preallocBlockSums(8192*8); //多めに取らないとアクセス違反が起こる
	cudaHostAlloc(&particle_set, 8192 * sizeof(float3), cudaHostAllocMapped); //ゼロコピーメモリ

	cudaMalloc(&importance,sizeof(float) * 8192);
	cudaMalloc(&importance_prefix,sizeof(float) * 8192);

	cudaHostAlloc(&__current_state, sizeof(float3), cudaHostAllocMapped); //ゼロコピーメモリ
	cudaHostAlloc(&lrf_host, sizeof(float2)*2000, cudaHostAllocMapped);

	cudaHostAlloc(&p, 8192 * sizeof(float), cudaHostAllocMapped); //ゼロコピーメモリ
	curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(g,clock());

	cudaHostGetDevicePointer(&lrf_device, lrf_host, 0);

	prescanArray(importance_prefix, importance, 16);

	//初期アンサンブル
	*__current_state=state;
	fill_particle<<<16,512>>>(*__current_state, particle_set);

	return 	__current_state;
}

int b_search(float ary[], float key, int imin, int imax) ;
float3 step(float3 current_speed) ////TODO:CUDAストリームの使用
{
	float3* dparticle;
	cudaHostGetDevicePointer(&dparticle, particle_set, 0);
	float3* cPos_device;
	cudaHostGetDevicePointer(&cPos_device, __current_state, 0);
	particle<<<16,512>>>(dparticle,importance , current_speed, cPos_device);
	prescanArray(importance_prefix, importance, 8192);
	int const n = 8192;
	float* dp;	cudaHostGetDevicePointer(&dp, p, 0);

	curandGenerateUniform(g, dp, n); //処理中に並行して抽選用乱数生成

	cudaDeviceSynchronize();


	float temp1;
	cudaMemcpy(&temp1, &importance_prefix[8191], sizeof(float),cudaMemcpyDeviceToHost);

	float max_imp;
	cudaMemcpy(&max_imp, &importance[8191], sizeof(float),cudaMemcpyDeviceToHost);

	max_imp+=temp1;

	float pref_host[8192];
	cudaMemcpy(pref_host, importance_prefix, sizeof(float) * 8192, cudaMemcpyDeviceToHost);

	float3 tmp_particle[8192];
#pragma omp parallel for
	for (int var = 0; var < 8192; ++var) {
		float random = p[var] * max_imp;
		int resample_index = b_search(pref_host,random,0,8191);
		tmp_particle[var] = particle_set[resample_index];
	}
	memcpy(particle_set,tmp_particle,8192 * sizeof(float3));



}

void Dispose()
{
	cudaFreeHost(p);
	cudaFreeHost(particle_set);
	cudaFreeHost(__current_state);
	cudaFree(importance);
	cudaFree(importance_prefix);
}
