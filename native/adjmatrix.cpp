/*
 * AdjMatrix.cpp
 *
 *  Created on: Jan 11, 2014
 *      Author: joe
 */

#include"adjmatrix.h"
#include<iostream>

template<>
void AdjMatrix<float>::floyd()
{
	cuTimer T;
	T.logstart();
	cuFloyd_F(cuMat,dim);
	T.logstop();
	CUDA_CHECK_RETURN(cudaGetLastError());
	T.logtime();
	std::cout<<T.gettime()<<std::endl;
	CUDA_CHECK_RETURN(cudaGetLastError());
}

template<>
void AdjMatrix<float>::fill()
{
	cuFill_F(cuMat,dim);
	CUDA_CHECK_RETURN(cudaGetLastError());
}

template<>
void AdjMatrix<int>::floyd()
{
	cuTimer T;
	T.logstart();
	cuFloyd_I(cuMat,dim);
	T.logstop();
	CUDA_CHECK_RETURN(cudaGetLastError());
	T.logtime();
	std::cout<<T.gettime()<<std::endl;
	CUDA_CHECK_RETURN(cudaGetLastError());
}

template<>
void AdjMatrix<double>::floyd()
{
	cuTimer T;
	T.logstart();
	cuFloyd_D(cuMat,dim);
	T.logstop();
	CUDA_CHECK_RETURN(cudaGetLastError());
	T.logtime();
	std::cout<<T.gettime()<<std::endl;
	CUDA_CHECK_RETURN(cudaGetLastError());
}

template<>
void AdjMatrix<int>::fill()
{
	cuFill_I(cuMat,dim);
	CUDA_CHECK_RETURN(cudaGetLastError());
}

template<>
void AdjMatrix<double>::fill()
{
	cuFill_D(cuMat,dim);
	CUDA_CHECK_RETURN(cudaGetLastError());
}
