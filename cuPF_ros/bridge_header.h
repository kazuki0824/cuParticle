/*
 * bridge_header.h
 *
 *  Created on: 2018/04/25
 *      Author: dvr1
 */

#ifndef BRIDGE_HEADER_H_
#define BRIDGE_HEADER_H_



//ところどころexternいらなさそうなので消してください

extern const int MAP_SIZE; //マップの一辺の長さ

__host__ void CopySensorData(float * from, size_t count);
__host__ void CopyMapData(float * from, size_t count);


/* 
float3* start(float)
初期の状態を入力すると初期アンサンブルを内部に生成します
戻されたポインタはホストメモリの上のパーティクル集合を格納する領域の先頭を指すポインタで、これを参照すればパーティクルそのものを可視化できる
第一引数: 初期状態
戻り値: パーティクル配列の先頭のポインタ
*/
float3* start(float x, float y, float z);

/*
float3 step(float3)
現在の速度を用いて一期先予測を行い、尤度を計算し、再サンプリングを行います
戻り値は最尤パーティクルそのものが値で帰ってくる
第一引数: 現在の移動速度
第二引数: 最尤粒子
*/
float3 step(float vx, float vy, float vz);

void Dispose();

#endif /* BRIDGE_HEADER_H_ */
