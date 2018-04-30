/*
 * likelihood.cu
 *
 *  Created on: 2018/04/21
 *      Author: kazuki
 */

#include <stdio.h>
extern __device__ float2* lrf_device;
extern __device__ float map[];
extern __constant__ size_t sensor_data_count;
extern const int MAP_SIZE; //マップの一辺の長さ
extern __constant__ int MAP_SIZE_DEVICE;

__device__ float likelihood(float3 state)
{
	const float map_real_width = 10.0;

	//TODO:ゆうど関数？
	//lrf_device[2000]を元に尤度を評価する

	//MAP_SIZE_DEVICE マップの"一辺"の長さ
	//sensor_data_count LRFのデータの総数
	//map デバイス側コンスタントメモリの上にコピーされたマップデータの配列。読み込み専用
	//lrf_device LRFのデータ。グローバルメモリの上にピンされたゼロコピーメモリの上に載っている。読み書き自由。

	//int thId=threadIdx.x+ blockDim.x * blockIdx.x; //今のパーティクルの番号
	int x_, y_;
	float l = 0.0;
#pragma unroll
	for(int i=0; i < sensor_data_count; i++)
	{
		x_ = MAP_SIZE_DEVICE /2 + (int)((state.x + lrf_device[i].x) * MAP_SIZE_DEVICE / map_real_width);
		y_ = MAP_SIZE_DEVICE /2 + (int)((state.y + lrf_device[i].y) * MAP_SIZE_DEVICE / map_real_width);

		if ((y_ * MAP_SIZE_DEVICE + x_)>=MAP_SIZE_DEVICE * MAP_SIZE_DEVICE || (y_ * MAP_SIZE_DEVICE + x_) <0)
		{
			printf("%d\n",y_ * MAP_SIZE_DEVICE + x_);
			return 0.0;
		}
		else
			l+=map[y_ * MAP_SIZE_DEVICE + x_];
	}


	return l;
}

