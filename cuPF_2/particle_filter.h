/*
 * particle_filter.h
 *
 *  Created on: 2018/10/11
 *      Author: dvr1
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_


#include "../modules/helper_math.h"
#include "../user/config.h"
#include "../user/behavior.h"

void Init(float x, float y, float z);
void Step(vParameter param);


#endif /* PARTICLE_FILTER_H_ */
