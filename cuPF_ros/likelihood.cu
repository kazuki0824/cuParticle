/*
 * likelihood.cu
 *
 *  Created on: 2018/04/21
 *      Author: kazuki
 */


extern __device__ float2* lrf_device;
extern __constant__ float map[];
extern __constant__ size_t sensor_data_count;
const int MAP_SIZE; //マップの一辺の長さ
__constant__ int MAP_SIZE_DEVICE = MAP_SIZE;

__forceinline__ __device__ float likelihood(float3 state)
{
	//TODO:ゆうど関数？
	//lrf_device[2000]を元に尤度を評価する

	//MAP_SIZE_DEVICE マップの"一辺"の長さ
	//sensor_data_count LRFのデータの総数
	//map デバイス側コンスタントメモリの上にコピーされたマップデータの配列。読み込み専用
	//lrf_device LRFのデータ。グローバルメモリの上にピンされたゼロコピーメモリの上に載っている。読み書き自由。

}

