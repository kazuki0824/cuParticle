/*
 * behavior.h
 *
 *  Created on: 2018/10/11
 *      Author: dvr1
 */

#ifndef BEHAVIOR_H_
#define BEHAVIOR_H_


static inline __device__ float2 prediction(float2 state, float3 mu, float sigma, curandState_t * s)
{
	// 姿勢の変量を平均、速度を標準誤差とする正規分布に従う乱数を生成する
	// 生成した乱数をパーティクルの移動量として加える
	// パーティクルの位置を返す

	float2 result;
	result.x = state.x + sigma * curand_normal(s) + mu.x;
	result.y = state.y + sigma * curand_normal(s) + mu.y;

	return result;
}

#endif /* BEHAVIOR_H_ */
