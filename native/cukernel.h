/*
 * cukernel.h
 *
 *  Created on: Jan 11, 2014
 *      Author: joe
 */

#ifndef CUKERNEL_H_
#define CUKERNEL_H_

#include"adjmatrix.h"

typedef struct
{
	unsigned int num;
	unsigned int stride;
} MatrixDim;

void cuFloyd_I(int* cuMat,MatrixDim dim);

void cuFloyd_F(float* cuMat,MatrixDim dim);

void cuFloyd_D(double* cuMat,MatrixDim dim);

void cuTransClosure(int* cuMat,MatrixDim dim);

void cuFill_F(float* cuMat,MatrixDim dim);

void cuFill_D(double* cuMat,MatrixDim dim);

void cuFill_I(int* cuMat,MatrixDim dim);

#endif /* CUKERNEL_H_ */
