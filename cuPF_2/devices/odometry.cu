#include "ros/ros.h"
#include "geometry_msgs/Pose2D.h"
#include "tf/transform_datatypes.h"
#include "tf/transform_broadcaster.h"
#include "helper_math.h"

#include "../particle_filter.h"

float3 state;       // オドメトリ空間上での姿勢
float3 diff;        // 姿勢の変化量
extern float3 pf_out_pose;

// パーティクルフィルタを再適用するために必要な移動量（位置、角度）
// 単位はそれぞれ[m]、[rad]
float threshold_transform = 0.05;
float threshold_rotate = 3.141592653589793f / 360;

// オドメトリ入力があったときの処理
void odometryCallback(const geometry_msgs::Pose2D& msg)
{
    /*
    float64 x
    float64 y
    float64 theta
    */
    static float3 cur_state;
    static float3 old_state;

    // 現在の姿勢情報を代入する
    cur_state.x = msg.x;
    cur_state.y = msg.y;
    cur_state.z = msg.theta;

    // 前回の姿勢情報取得時からの差分を得る
    diff = cur_state - old_state;

    // 差分が閾値を超えていた場合、パーティクルフィルタを適用する
    if(pow(diff.x, 2) + pow(diff.y, 2) > pow(threshold_transform, 2) ||
       diff.z > threshold_rotate)
    {
        // パーティクルフィルタ呼び出し
        Step();
        // 姿勢推定結果をTFとして配信する
        static tf::TransformBroadcaster br;
        tf::Transform transform;
        tf::Quaternion q;

        q.setRPY(0, 0, pf_out_pose.z);
        transform.setOrigin(tf::Vector3(pf_out_pose.x - state.x, pf_out_pose.y - state.y, 0));
        transform.setRotation(q);

        br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "odom"));
    }
}
