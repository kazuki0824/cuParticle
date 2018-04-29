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
	float xx[5] = {0.0,0.1,0.2};
	float2 x[4] ;
	float3 state, velocity;
	start(state);
	CopyMapData(xx,3);
	CopySensorData(x,2);
	step(velocity);
}
