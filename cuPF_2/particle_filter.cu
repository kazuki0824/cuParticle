/*
 * particle_filter.cu
 *
 *  Created on: 2018/10/11
 *      Author: maleicacid
 */


#include <cub/device/device_scan.cuh>
#include <curand.h>
#include <curand_kernel.h>

#include "particle_filter.h"
#include "user/likelihood.h"

#include <stdio.h>
int b_search(float ary[], float key, int imin, int imax) {
    if (imax < imin) {
        return imax;
    } else {
        int imid = imin + (imax - imin) / 2;
        if (ary[imid] > key) {
            return b_search(ary, key, imin, imid - 1);
        } else if (ary[imid] < key) {
            return b_search(ary, key, imid + 1, imax);
        } else {
            return imid;
        }
    }
}


/*********************************************************/

float3 state = {0};
static void SetupLMap() //尤度マップ転送
{

}


float * p;
curandGenerator_t g;
void Init(float x, float y, float z)
{
	state = make_float3(x,y,z);

	SetupLMap();

	//Init RNG (Host)
	curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(g,clock());

}
void Step(vParameter param)
{
	cudaHostAlloc(&p, 8192 * sizeof(float), cudaHostAllocMapped); //ゼロコピーメモリ

	//Generate random values for re-sampling
	float* dp; cudaHostGetDevicePointer(&dp, p, 0);
	curandGenerateUniform(g, dp, sample_count);


}
