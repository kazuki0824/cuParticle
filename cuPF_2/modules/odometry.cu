#include "ros/ros.h"
#include "geometry_msgs/Pose.h"
#include "tf/transform_datatypes.h"

#include "../particle_filter.h"

float3 state;

void poseCallback(const geometry_msgs::Pose& msg)
{
    /*
    Point position
        float64 x
        float64 y
        float64 z
    Quaternion orientation
        float64 x
        float64 y
        float64 z
        float64 w
    */
    float3 old_state = state;
    float3 diff;

    // パーティクルフィルタを再適用するために必要な移動量（位置、角度）
    // 単位はそれぞれ[m]、[rad]
    float thr_trans = 0.05;
    float thr_rot = 3.141592653589793f / 360;

    // 現在の姿勢情報を代入する
    state.x = msg.x;
    state.y = msg.y;

    // 前回の姿勢情報取得時からの差分を得る
    diff = state - old_state;

    // 差分が閾値を超えていた場合、パーティクルフィルタを適用する
    if(pow(diff.x, 2) + pow(diff.y, 2) > pow(thr_trans, 2) ||
       diff.z > thr_rot)
    {
        // パーティクルフィルタ呼び出し
        Step()
    }
}