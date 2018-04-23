/*
 * main.cu
 *
 *  Created on: 2018/04/23
 *      Author: dvr1
 */


#include <stdio.h>
#include "scan_largearray_kernel.h"
#include "kernel.hcu"


int main()
{
	preallocBlockSums(20000);
	float * x;
	float * y;
	cudaMalloc(&x,20000*4);
	cudaMalloc(&y,20000*4);

	float xx[256]= {1.0,2.0,3.0};
	cudaMemcpy(x,xx,12,cudaMemcpyHostToDevice);
	prescanArray(y,x,1024);
	cudaMemcpy(xx,y,16,cudaMemcpyDeviceToHost);


	printf("%f,%f,%f,%f,",xx[0],xx[1],xx[2],xx[3]);

	start();
	step();
}
