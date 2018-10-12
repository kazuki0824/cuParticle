/*
 * PF.cpp
 *
 *  Created on: 2018/10/12
 *      Author: dvr1
 */

/*
 * nvccで２つ以上のコンパイル単位でpcl/point_cloud.hをインクルードすると、
 * Eigenが何らかの干渉を起こしてリンク時に多重定義のエラーになる
 * よって、Pointcloud系だけここに書く。名前空間pclの型はcudaへ公開してはいけない
 */
#include "modules/emicp.h"
