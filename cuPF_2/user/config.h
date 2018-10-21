/*
 * config.h
 *
 *  Created on: 2018/10/11
 *      Author: maleicacid
 */

#ifndef CONFIG_H_
#define CONFIG_H_


const int sample_count = 8192;

const int MAP_SIZE = 1024;

// ビームモデル用のパラメーター
#define z_hit 1.0 //TODO:
#define z_short 1.0 //TODO:
#define hit 1.0 //TODO:
#define z_hit 1.0 //TODO:
#define z_ 1.0 //TODO:
#define z_max 1.0 //TODO:

struct vParameter
{
	float vx;
	float vy;
};


static vParameter default_param;


#endif /* CONFIG_H_ */
