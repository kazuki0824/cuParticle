/*
 * likelihood.cu
 *
 *  Created on: 2018/04/21
 *      Author: kazuki
 */


float2* lrf_device;
__host__ inline float2* getSensorDataPtr(){
	cudaHostGetDevicePointer(&lrf_device, lrf_host, 0);
	return lrf_host;
}

__device__ float likelihood(float3 state)
{
	//TODO:ゆうど関数？
	//lrf_device[2000]を元に尤度を評価する
}

