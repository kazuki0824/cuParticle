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

/**************************************************************************************************************
 *
 * https://github.com/v-mehta/cuda_emicp_softassign#how-to-use-the-apiを参考に、EM-ICPを呼び出すことが目的。
 * これをする前にLRFの生データをpcl::PointCloud<pcl::PointXYZ>::Ptrに書き換える前処理をperformICP(void *)に実装する。
 *
 */

void performICP(void * hLRF)
{
	//TODO: 各引数を用意(LRFを読んでPCLのかたちへ変換)↓


	//TODO: emicp呼出↓
	//emicp(cloud_targetX, cloud_sourceY , float* h_R, float* h_t , param)の形。
	//paramは、registrationParameterという構造体で、emicp.hに定義がある。
	//意味はhttps://github.com/v-mehta/cuda_emicp_softassign#how-to-use-the-apiの通り。
	//パラメータの意味はhttps://ieeexplore.ieee.org/abstract/document/5695231にあるらしい(?)

}
