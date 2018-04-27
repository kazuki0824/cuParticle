/*
 * bridge_header.h
 *
 *  Created on: 2018/04/25
 *      Author: dvr1
 */

#ifndef BRIDGE_HEADER_H_
#define BRIDGE_HEADER_H_


__host__ void CopySensorData(float2 * from, size_t count);
extern float3* start(float3 state);
extern float3 step(float3 current_speed);


#endif /* BRIDGE_HEADER_H_ */
