#include "ros/ros.h"
#include "geometry_msgs/Pose.h"
#include "tf/transform_datatypes.h"
#include "helper_math.h"

#include "../particle_filter.h"

float3 state;

// パーティクルフィルタを再適用するために必要な移動量（位置、角度）
// 単位はそれぞれ[m]、[rad]
float threshold_transform = 0.05;
float threshold_rotate = 3.141592653589793f / 360;

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

    // 現在の姿勢情報を代入する
    state.x = msg.position.x;
    state.y = msg.position.y;

    // 前回の姿勢情報取得時からの差分を得る
    diff = state - old_state;

    // 差分が閾値を超えていた場合、パーティクルフィルタを適用する
    if(pow(diff.x, 2) + pow(diff.y, 2) > pow(threshold_transform, 2) ||
       diff.z > threshold_rotate)
    {
        // パーティクルフィルタ呼び出し
        Step();
    }

}
