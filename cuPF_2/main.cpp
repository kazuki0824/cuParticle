/*
 * main.c
 *
 *  Created on: 2018/10/11
 *      Author: maleicacid
 */

#include "ros/ros.h"

#include "particle_filter.h"

int main(int argc, char** argv)
{
	ros::init(argc, argv, "cu_particle");
	ros::NodeHandle nh("~");

	Init(0,0);
	// 配信するトピックを宣伝する

	// 購読するトピックとコールバック関数を登録する

	// これ以降メイン関数はトピックに反応する以外特に何もしない
	ros::spin();
}
