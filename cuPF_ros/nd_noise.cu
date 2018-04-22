/*
 * nd_noise.cu
 *
 *  Created on: 2018/04/21
 *      Author: kazuki
 */

#include <curand_kernel.h>




__device__ static float noisegen(curandState_t * s, float mean = 0.0, float stddev =1.0)
{
	return stddev*curand_normal(s) + mean;
}

__device__ float3 getSystemNoise(curandState_t * s)
{

	struct mean_var
	{
		float x_mean = 0.0;
		float x_stddev = 0.0;
		float y_mean = 0.0;
		float y_stddev = 0.0;
		float theta_mean = 0.0;
		float theta_stddev = 0.0;
	};
	mean_var param;
	float3 v = { noisegen(s,param.x_mean,param.x_stddev), noisegen(s,param.y_mean,param.y_stddev), noisegen(s,param.theta_mean,param.theta_stddev) };
	return v;
}

__device__ float3 suggest_distribution(float3 x, float3 system_noise)
{
	//TODO:状態遷移関数?
	return x + system_noise;
}

