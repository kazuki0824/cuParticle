/*
 * main.cu
 *
 *  Created on: 2018/04/23
 *      Author: dvr1
 */


#include <stdio.h>
#include "scan_largearray_kernel.h"
#include "kernel.hcu"
#include "bridge_header.h"

int main()
{
	float xx[8] = {0.0,0.1,0.2};
	float x[4] ;
	float3 state, velocity;
	start(x[0],x[1],x[2]);
	CopyMapData(xx,3);
	CopySensorData(x,2);
	float3 data = step(x[0],x[1],x[2]);
	printf("%f,%f,%f\n",data.x,data.y,data.z);
}
