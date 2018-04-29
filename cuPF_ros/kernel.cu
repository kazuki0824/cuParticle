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


float2* lrf_host;
__device__ float2* lrf_device;


__constant__ size_t sensor_data_count;
__host__ void CopySensorData(float2 * from, size_t count)
{
	memcpy(lrf_host,from,count);

	cudaMemcpyToSymbol(sensor_data_count, &count, sizeof(size_t));
}


const int MAP_SIZE = 1024;
__device__ float map[MAP_SIZE*MAP_SIZE];
__host__ void CopyMapData(float * from, size_t count)
{
	cudaMemcpyToSymbol(map, from, count);
}


__device__ float pred(float3 * current, float3 current_speed)
{
	unsigned int seed = current_speed.x;
	unsigned int id = current_speed.y;
	curandState_t s;
	curand_init(seed, id, 0, &s);//高速化を図りたい


	float3 prediction = suggest_distribution(*current ,current_speed, &s);

	float l = likelihood(prediction); //予測のもっともらしさを計算する

	*current = prediction;

	return l;

}

__global__ void fill_particle(float3 initialState, float3 * p)
{
	unsigned int seed = initialState.x;
	unsigned int id = initialState.y;
	curandState_t s;
	curand_init(seed, id, 0, &s);
	int thId=threadIdx.x+ blockDim.x * blockIdx.x;

	mean_var_param pa;
	pa.x_stddev = pa.y_stddev = 1.0;
	pa.theta_stddev = 0.4;
	p[thId] = initialState + getSystemNoise(&s, pa);
}

__global__ void particle(float3 *particles, float *importance, float3 current_speed, float3 *__state_ptr)
{
	__shared__ float max_per_blocks[16];
	__shared__ int argmax_per_blocks[16];

	int thId=threadIdx.x+ blockDim.x * blockIdx.x;
	float value2 = importance[thId] = pred(&particles[thId], current_speed);
	float value = value2;
	//並列リダクションで最大値を求める

	//各ワープ内部での評価
	for (int i=1; i<32; i*=2)
		value = fmaxf(value, __shfl_xor(-1, value, i)); //バタフライ交換の要領
	if(value2 == value)//この条件にマッチするのはワープごとに一つのスレッドだけ
	{
		max_per_blocks[threadIdx.x/32] = value;
		argmax_per_blocks[threadIdx.x/32] = thId;
	}
	__syncthreads();

	//ブロック単位の評価
	float* ptr = (float*)malloc(sizeof(float)*16);
	int argmax_in_this_block =0;
	if(threadIdx.x<8)
	{
		value = value2 = max_per_blocks[threadIdx.x];
		for (int i=1; i<8; i*=2)
			value = fmaxf(value, __shfl_xor(-1, value, i)); //バタフライ交換の要領

		if(value2 == value)
		{
			ptr[blockIdx.x]=value;
			argmax_in_this_block = argmax_per_blocks[threadIdx.x];
		}

		__syncthreads();

		//ブロック間の評価
		int block_with_max =0;
		if(blockIdx.x==0)
		{
			float * final_temp = max_per_blocks;

			final_temp[threadIdx.x] = ptr[threadIdx.x];
			value = value2 = final_temp[threadIdx.x];
			for (int i=1; i<8; i*=2)
				value = fmaxf(value, __shfl_xor(-1, value, i)); //バタフライ交換の要領
			if(value2 == value)
			{
				block_with_max = threadIdx.x;
			}
		}
		if((blockIdx.x==block_with_max) && (threadIdx.x==0)){
			*__state_ptr=particles[argmax_in_this_block];
		}
	}
	free(ptr);
}


float3 * particle_set;
float * importance;
float * importance_prefix;

curandGenerator_t g;
float* p;

float3* __current_state;
__constant__ int MAP_SIZE_DEVICE;
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

	float2* lrf_device_;
	cudaHostGetDevicePointer(&lrf_device_, lrf_host, 0);
	cudaMemcpyToSymbol(lrf_device,&lrf_device_,sizeof(float2*));

	prescanArray(importance_prefix, importance, 8);

	//初期アンサンブル
	*__current_state=state;

	float3* dparticle;
	cudaHostGetDevicePointer(&dparticle, particle_set, 0);

	fill_particle<<<128,64>>>(*__current_state, dparticle);
	cudaMemcpyToSymbol(MAP_SIZE_DEVICE, &MAP_SIZE, sizeof(int));

	return 	particle_set;
}

int b_search(float ary[], float key, int imin, int imax) ;
float3 step(float3 current_speed) ////TODO:CUDAストリームの使用
{
	float3* dparticle;
	cudaHostGetDevicePointer(&dparticle, particle_set, 0);
	float3* cPos_device;
	cudaHostGetDevicePointer(&cPos_device, __current_state, 0);
	particle<<<16,256>>>(dparticle,importance , current_speed, cPos_device);
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

	return *__current_state;

}

void Dispose()
{
	cudaFreeHost(p);
	cudaFreeHost(particle_set);
	cudaFreeHost(__current_state);
	cudaFreeHost(lrf_host);
	cudaFree(importance);
	cudaFree(importance_prefix);
}
