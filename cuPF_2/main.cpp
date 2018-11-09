/*
 * main.c
 *
 *  Created on: 2018/10/11
 *      Author: maleicacid
 */

#include "ros/ros.h"

#include "particle_filter.h"
#include "sensor_msgs/LaserScan.h"
#include "geometry_msgs/Pose2D.h"
#include "nav_msgs/OccupancyGrid.h"

// コールバック関数
void scanCallback(const sensor_msgs::LaserScan&);
void odometryCallback(const geometry_msgs::Pose2D&);
void mapCallback(const nav_msgs::OccupancyGrid&);

int main(int argc, char** argv)
{
	ros::init(argc, argv, "cu_particle");
	ros::NodeHandle nh("~");

	Init(0,0);

	// 姿勢情報はこのノードからは配信しない（代わりにtfを配信する）
	// 購読するトピックとコールバック関数を登録する
	ros::Subscriber odomSub = nh.subscribe("robot_pose", 10, odometryCallback);
	ros::Subscriber scanSub = nh.subscribe("scan", 10, scanCallback);
	ros::Subscriber mapSub = nh.subscribe("map", 1, mapCallback);

	// これ以降メイン関数はトピックに反応する以外特に何もしない
	ros::spin();
}
