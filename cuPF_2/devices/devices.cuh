#ifndef DEF_DEVICES_H

#define DEF_DEVICES_H

#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"
#include "geometry_msgs/Pose.h"

// コールバック関数
void scanCallback(const sensor_msgs::LaserScan&);
void poseCallback(const geometry_msgs::Pose&);

// パラメーター群
extern float threshold_transform;
extern float threshold_rotate;

// LRF用のデータ
extern int nBeam;
extern float2 *hLRF;

#endif