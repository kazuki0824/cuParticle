#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"

#include "particle_filter.h"
#include "../modules/emicp.cuh"
#include "../modules/emicp.h"

int nBeam;
float2 * hLRF;
extern float3 state;

// スキャンデータを受け取ったときのコールバック関数
void scanCallback(const sensor_msgs::LaserScan& msg)
{
	/*
	float angle_min;       	start angle of the scan [rad]
	float angle_max;       	end angle of the scan [rad]
	float angle_increment; 	angular distance between measurements [rad]
	float time_increment;  	time between measurements [seconds] - if your scanner
                       		is moving, this will be used in interpolating position
                       		of 3d points
	float scan_time;       	time between scans [seconds]
	float range_min;       	minimum range value [m]
	float range_max;       	maximum range value [m]
	float* ranges;        	range data [m] (Note: values < range_min or > range_max should be discarded)
	float* intensities;
	*/
	static int nScan = 0;
	nScan++;
	nBeam = (int)((msg.angle_max - msg.angle_min) / msg.angle_increment + 1);

	for(int i = 0; i < nBeam; i++)
	{
		hLRF[i].x = ranges[i];
		hLRF[i].y = msg.angle_min + (msg.angle_increment * i);
	}

	// 二次元点群形式に変換する
	static float3 cloud[2][2048];
	static int size_cloud[2];
	static int current_cloud = 0;

	size_cloud[current_cloud] = nBeam;
	for(int i=0; i < nBeam; i++)
	{
		cloud[current_cloud][i].x = hLRF[i].x * cos(hLRF[i].y);
		cloud[current_cloud][i].y = hLRF[i].x * sin(hLRF[i].y);
	}
	current_cloud = (current_cloud + 1) % 2;

	// スキャン回数が2回以上であった場合、前回のものと比較してICPを実行する
	if(nScan > 1)
	{
		float R[] = {
			1.0, 0.0, 0.0, 
			0.0, 1.0, 0.0, 
			0.0, 0.0, 1.0
		};
		float t[] = {0.0, 0.0, 0.0};
		registrationParameters param;
		param.sigma_p2 = 0.01;
		param.sigma_inf = 0.00001;
		param.sigma_factor = 0.9;
		param.d_02 = 0.01;

		emicp(cloud[current_cloud], cloud[(current_cloud + 1) % 2], 
			size_cloud[current_cloud], size_cloud[(current_cloud + 1) % 2, R, t, param);
		state.z = acos(R[8]);
	}
}