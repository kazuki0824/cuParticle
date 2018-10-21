/*
  Copyright (c) 2014 Toru Tamaki

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
#include "emicp.h"

// h_X stores points as the order of
// [X_x1 X_x2 .... X_x(Xsize-1) X_y1 X_y2 .... X_y(Xsize-1)  X_z1 X_z2 .... X_z(Xsize-1) ],
// where (X_xi X_yi X_zi) is the i-th point in X.
// h_Y does the same for Y.
void cloud2data(const float2 cloud, float **X, int &Xsize) {
	float* h_X = new float [Xsize * 3];
	float* h_Xx = &h_X[Xsize*0];
	float* h_Xy = &h_X[Xsize*1];
	float* h_Xz = &h_X[Xsize*2];

	for (int i = 0; i < Xsize; i++) {
		h_Xx[i] = cloud[i].x;
		h_Xy[i] = cloud[i].y;
		h_Xz[i] = 0;
	}
	*X = h_X;
}
