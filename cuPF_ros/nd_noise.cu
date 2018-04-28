/*
 * nd_noise.cu
 *
 *  Created on: 2018/04/21
 *      Author: kazuki
 */

#include <curand_kernel.h>
#include "helper_math.h"

#include "nd_noise.hcu"

__device__ static float noisegen(curandState_t * s, float mean = 0.0, float stddev =1.0)
{
	return stddev*curand_normal(s) + mean;
}

__device__ float3 getSystemNoise(curandState_t * s, mean_var_param param)
{
	float3 v = { noisegen(s,param.x_mean,param.x_stddev), noisegen(s,param.y_mean,param.y_stddev), noisegen(s,param.theta_mean,param.theta_stddev) };
	return v;
}

__device__ float3 suggest_distribution(float3 x, float3 speed, curandState_t *s)
{
	//TODO:状態遷移関数?

	//int thId=threadIdx.x+ blockDim.x * blockIdx.x; //今のパーティクルの番号
	float delta_rot1, delta_trans, delta_rot2;
	float delta_rot1_, delta_trans_, delta_rot2_;
	float alpha[4] = {1.0, 1.0, 1.0, 1.0};
	float3 result;

	delta_rot1 = atan2f(speed.y, speed.x) - x.z;
	delta_trans = sqrtf(speed.y * speed.y + speed.x * speed.x);
	delta_rot2 = speed.z - delta_rot1;

	delta_rot1_  = (delta_rot1
					- (  alpha[0] * powf(delta_rot1, 2)
					   + alpha[1] * powf(delta_trans, 2)
					  )
				   ) * curand_normal(s);
	delta_trans_ = (delta_trans
					- (  alpha[2] * powf(delta_trans, 2)
					   + alpha[3] * powf(delta_rot1, 2)
					   + alpha[3] * powf(delta_rot2, 2)
					  )
				   ) * curand_normal(s);
	delta_rot2_  = (delta_rot2
					- (  alpha[0] * powf(delta_rot2, 2)
					   + alpha[1] * powf(delta_trans, 2)
					  )
				   ) * curand_normal(s);
	result.x = x.x + delta_trans_ * cosf(x.z + delta_rot1_);
	result.y = x.y + delta_trans_ * sinf(x.z + delta_rot1_);
	result.z = x.z + delta_rot1_ + delta_rot2_;

	return result;
}

