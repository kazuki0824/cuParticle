/*
  Copyright (c) 2010 Toru Tamaki

  Permission is hereby granted, free of charge, to any person
  obtaining a copy of this software and associated documentation
  files (the "Software"), to deal in the Software without
  restriction, including without limitation the rights to use,
  copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the
  Software is furnished to do so, subject to the following
  conditions:

  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
  OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
  HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
  OTHER DEALINGS IN THE SOFTWARE.
*/
#pragma once

#include <pcl/common/transforms.h>

/// Struct containing the parameters to do the point cloud registration.
typedef struct {
	// EM-ICP parameters
	float sigma_p2;      // initial value for the main loop. sigma_p2 <- sigma_p2 * sigma_factor  at the end of each iteration while sigma_p2 > sigam_inf. default: 0.01
	float sigma_inf;     // minimum value of sigma_p2. default: 0.00001
	float sigma_factor;  // facfor for reducing sigma_p2. default: 0.9
	float d_02;          // values for outlier (see EM-ICP paper). default: 0.01

	// misc
	int notimer;  // No timer is shown.

} registrationParameters;

void emicp(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target, const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, float* h_R, float* h_t, const registrationParameters &param);

// Find the transformation matrix from the scene.
void findRTfromS(const float* h_Xc, const float* h_Yc, const float* h_S, float* h_R, float* h_t);

void cloud2data(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float **X, int &Xsize);

