/*
 * binary_search.cpp
 *
 *  Created on: 2018/04/24
 *      Author: dvr1
 */

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
