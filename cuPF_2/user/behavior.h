/*
 * behavior.h
 *
 *  Created on: 2018/10/11
 *      Author: dvr1
 */

#ifndef BEHAVIOR_H_
#define BEHAVIOR_H_

static inline __device__ float getNormal(float sigma, curandState_t * s)
{
	return sigma * curand_normal(s);
}

static inline __device__ float2 prediction(float2 state, vParameter p, curandState_t * s)
{
	//TODO:
	return state;
}





#endif /* BEHAVIOR_H_ */
