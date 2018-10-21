/*
 * cub_wrapper.h
 *
 *  Created on: 2018/10/11
 *      Author: dvr1
 */

#ifndef CUB_WRAPPER_H_
#define CUB_WRAPPER_H_


#include <cub/device/device_scan.cuh>


static void incl_scan_CDF(float * d_in,float * h_out,int num_items,cudaStream_t * stream)
{
/*
	int *d_in = NULL;
	CubDebugExit(
			cudaMalloc((void**) &d_in,
					sizeof(float) * num_items));

	// Initialize device input
	CubDebugExit(
			cudaMemcpy(d_in, h_in, sizeof(int) * num_items,
					cudaMemcpyHostToDevice));
*/
	// Allocate device output array
	int *d_out = NULL;
	CubDebugExit(
			cudaMalloc((void**) &d_out,
					sizeof(int) * num_items));

	// Determine temporary device storage requirements for inclusive prefix scan
	void *d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, *stream);
	// Allocate temporary storage for inclusive prefix scan
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	// Run inclusive prefix sum scan
	cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, *stream);

	CubDebugExit(
				cudaMemcpy(h_out, d_out, sizeof(int) * num_items,
						cudaMemcpyDeviceToHost));

	//cudaFree(d_in);
	cudaFree(d_out);
	cudaFree(d_temp_storage);
}


#endif /* CUB_WRAPPER_H_ */
