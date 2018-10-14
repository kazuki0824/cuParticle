#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"

#include "particle_filter.h"

int nBeam;
float2 * hLRF;

void scanCallback(const sensor_msgs::LaserScan& msg)
{
	//float angle_min;       	 start angle of the scan [rad]
	//float angle_max;       	 end angle of the scan [rad]
	//float angle_increment; 	 angular distance between measurements [rad]
	//float time_increment;  	 time between measurements [seconds] - if your scanner
    //                   		 is moving, this will be used in interpolating position
    //                   		 of 3d points
	//float scan_time;       	 time between scans [seconds]
	//float range_min;       	 minimum range value [m]
	//float range_max;       	 maximum range value [m]
	//float* ranges;        	 range data [m] (Note: values < range_min or > range_max should be discarded)
	//float* intensities;

	int nBeam = (int)((msg.angle_max - msg.angle_min)/msg.angle_increment + 1);

	for(int i = 0; i < nBeam; i++)
	{
        float angle = msg.angle_min + (msg.angle_increment * i);
        hLRF[i].x = ranges[i] * cos(angle);
        hLRF[i].y = ranges[i] * sin(angle);
	}
}