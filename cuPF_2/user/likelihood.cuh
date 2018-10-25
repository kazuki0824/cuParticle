/*
 * likelihood.h
 *
 *  Created on: 2018/10/11
 *      Author: dvr1
 */

#ifndef LIKELIHOOD_H_
#define LIKELIHOOD_H_

#include "../inc/helper_math.h"
#include "user/config.h"

// 地図上のある点が占有されているかどうかを判定する
// 占有または地図外参照で1.0を、それ以外は0.0を返す。
__device__ inline float
isCellOccupied(float2 point, float* map_device, int width, int height, float resolution, float2 center)
{
    int2 iPoint;
    iPoint.x = (point.x + center.x) / resolution;
    iPoint.y = (point.y + center.y) / resolution;
    if(iPoint.x < 0 || iPoint.y < 0 || iPoint.x > width - 1 || iPoint.y > height - 1)
        return 1.0;
    return map_device[iPoint.y * width + iPoint.x];
}

// 尤度を計算する
// old_likelihood : そのパーティクルの前回の尤度
// state : そのパーティクルの位置
// LRF_device : その時点でのスキャンデータ
// map_device : 地図データ
static __device__ float 
likelihood(float old_likelihood, float2 state, float2* LRF_device, int nBeam, 
           float* map_device, int width, int height, float resolution, float2 center)
{
    // 各レーザー方向の最も近い点を計算する
    float2 step;
    float2 est_endpoint = state;
    float est_z;
    float act_z;
    float result = 0;

    // ビームごとに尤度を計算し、集計する
    for(int i = 0; i < nBeam; i++)
    {
        step.x = 1.0 * cos(LRF_device[i].y);
        step.y = 1.0 * sin(LRF_device[i].y);
        // 終点を壁にぶつかるまで伸ばす
        while(isCellOccupied(est_endpoint, map_device, width, height, resolution, center) == 0.0)
        {
            est_endpoint += step;
        }
        est_z = length(est_endpoint - state);
        act_z = LRF_device[i].x;

        // 計測距離の誤差は正規分布に従うものとして考える
        result += 1/sqrt(2*3.141592653589793f*(z_hit))*exp(-pow((act_z)-(est_z), 2)/(2*pow((z_), 2))*hit);

        // 予期せぬ障害物の存在を考慮する
        if(act_z < est_z)
        {
            result += z_short / (1.0f - exp(-z_short * est_z)) * exp(-z_short * est_z);
        }

        // 計測失敗の可能性を考慮する
        if(act_z > z_max){
            result += 1.0;
        }
    }
    return result;
}


#endif /* LIKELIHOOD_H_ */
