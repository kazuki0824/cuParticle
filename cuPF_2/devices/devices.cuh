#ifndef DEF_DEVICES_H

#define DEF_DEVICES_H

#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"
#include "geometry_msgs/Pose.h"
#include "nav_msgs/OccupancyGrid.h"

// コールバック関数
void scanCallback(const sensor_msgs::LaserScan&);
void poseCallback(const geometry_msgs::Pose&);
void mapCallback(const nav_msgs::OccupancyGrid&);

// パラメーター群
extern float threshold_transform;
extern float threshold_rotate;

// LRF用のデータ
extern int nBeam;
extern float2 *hLRF;

// 地図の情報
extern int width;
extern int height;
extern float resolution;
extern float2 center;
extern float* h_map;        // ホスト側で読み込んだ地図（占有格子）
extern float* d_map;        // デバイス側に転送された地図

#endif