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
/*
struct vParameter
{

};
*/
typedef float2 vParameter; // 速度情報だけじゃ物足りないなら、上の構造体を使う
static vParameter default_param;

#endif /* CONFIG_H_ */
