/*
 * cukernels.c
 *
 *  Created on: Jan 11, 2014
 *      Author: joe
 */
#include"adjmatrix.h"
#include"cukernels.h"
#include<cmath>

template<typename T>
__global__ void cuFloydIDKnl(T *adjmat,const MatrixDim dim,const unsigned int node)
{
	unsigned int x=threadIdx.x,y=threadIdx.y;
	unsigned int pos=(node+y)*dim.stride+node+x;
	__shared__ T tmpL[4][16][16],tmpR[16][64];
	T len[4];
	T nlen;
	for (int i=0;i<4;++i)
		len[i]=adjmat[pos+((i*dim.stride)<<4)];
	for (int i=0;i<4;++i)
	for (int k=0;k<16;++k)
	{
		tmpR[y][x]=len[i];
		if ((x>>4)==i)
			for (int j=0;j<4;++j)
				tmpL[j][y][x&15]=len[j];
		__syncthreads();
		for (int j=0;j<4;++j)
		{
			nlen=tmpL[j][y][k]+tmpR[k][x];
			len[j]=len[j]>nlen?nlen:len[j];
		}
		__syncthreads();
	}
	for (int i=0;i<4;++i)
		adjmat[pos+((i*dim.stride)<<4)]=len[i];
}

template<typename T>
__global__ void cuFloydSDKnl(T *adjmat,const MatrixDim dim,const unsigned int node)
{
	unsigned int x=threadIdx.x,y=threadIdx.y,ry,rx;
	ry=y&15;
	rx=(y&0xFFF0)+x;
	unsigned int bx=blockIdx.x,by=blockIdx.y;
	unsigned int posn,pos;
	__shared__ T tmpL[64][16],tmpR[16][64];
	T len[4];
	T nlen;
	if (by==0)
	{
		posn=(node+y)*dim.stride+node+x;
		pos=(node+ry)*dim.stride+bx*ITERSIZE+rx;
		for (int i=0;i<4;++i)
			len[i]=adjmat[pos+((i*dim.stride)<<4)];
		for (int i=0;i<4;++i)
		{
			tmpL[y][x]=adjmat[posn+(i<<4)];
			for (int k=0;k<16;++k)
			{
				//if (ry==k)
				tmpR[ry][rx]=len[i];
				__syncthreads();
				for (int j=0;j<4;++j)
				{
					nlen=tmpL[(j<<4)+ry][k]+tmpR[k][rx];
					len[j]=len[j]>nlen?nlen:len[j];
				}
				__syncthreads();
			}

		}
		for (int i=0;i<4;++i)
			adjmat[pos+((i*dim.stride)<<4)]=len[i];
	}
	else
	{
		posn=(node+ry)*dim.stride+node+rx;
		pos=(bx*ITERSIZE+y)*dim.stride+node+x;
		for (int i=0;i<4;++i)
			len[i]=adjmat[pos+(i<<4)];
		for (int i=0;i<4;++i)
		{
			tmpR[ry][rx]=adjmat[posn+((i*dim.stride)<<4)];
			for (int k=0;k<16;++k)
			{
				//if (x==k)
				tmpL[y][x]=len[i];
				__syncthreads();
				for (int j=0;j<4;++j)
				{
					nlen=tmpL[y][k]+tmpR[k][(j<<4)+x];
					len[j]=len[j]>nlen?nlen:len[j];
				}
				__syncthreads();
			}
			//__syncthreads();
		}
		for (int i=0;i<4;++i)
			adjmat[pos+(i<<4)]=len[i];
	}
}

template<typename T>
__global__ void
__launch_bounds__(1024, 2)
cuFloydDDKnl(T *adjmat,const MatrixDim dim,const unsigned int node)
{
	unsigned int x=threadIdx.x,y=threadIdx.y,ry,rx;
	ry=y&15;
	rx=(y&0xFFF0)+x;
	unsigned int bx=blockIdx.x,by=blockIdx.y;
	unsigned int posl=(by*ITERSIZE+y)*dim.stride+node+x
				,posr=(node+ry)*dim.stride+bx*ITERSIZE+rx
				,pos=(by*ITERSIZE+y)*dim.stride+bx*ITERSIZE+x;
	__shared__ T tmpL[ITERSIZE][16],tmpR[16][ITERSIZE];
	T len[4];
	T nlen;
	for (int i=0;i<4;++i)
		len[i]=adjmat[pos+(i<<4)];
	for (int i=0;i<4;++i)
	{
		tmpL[y][x]=adjmat[posl];
		posl+=16;
		tmpR[ry][rx]=adjmat[posr];
		posr+=(dim.stride<<4);
		__syncthreads();
		for (int j=0;j<4;++j)
		for (int k=0;k<16;++k)
		{
			nlen=tmpL[y][k]+tmpR[k][(j<<4)+x];
			len[j]=len[j]>nlen?nlen:len[j];
		}
		__syncthreads();
	}
	for (int i=0;i<4;++i)
		adjmat[pos+(i<<4)]=len[i];
}

template<typename T>
__global__ void cuFillKnl(T *adjmat,const MatrixDim dim,const T maxval)
{
	unsigned int row=blockIdx.y;
	unsigned int col=ITERSIZE*blockIdx.x+threadIdx.x;
	if (col>=dim.num || row>=dim.num)
		adjmat[row*dim.stride+col]=maxval;
}

void cuFloyd_I(int* cuMat,MatrixDim dim)
{
	for (int k=0;k<dim.stride;k+=ITERSIZE)
	{
		cuFloydIDKnl<<<1,dim3(ITERSIZE,16,1)>>>(cuMat,dim,k);
		cuFloydSDKnl<<<dim3(dim.stride/ITERSIZE,2,1),dim3(16,ITERSIZE,1)>>>(cuMat,dim,k);
		cuFloydDDKnl<<<dim3(dim.stride/ITERSIZE,dim.stride/ITERSIZE,1),dim3(16,ITERSIZE,1)>>>(cuMat,dim,k);
	}
}

void cuFloyd_F(float* cuMat,MatrixDim dim)
{
	for (int k=0;k<dim.stride;k+=ITERSIZE)
	{
		cuFloydIDKnl<<<1,dim3(ITERSIZE,16,1)>>>(cuMat,dim,k);
		cuFloydSDKnl<<<dim3(dim.stride/ITERSIZE,2,1),dim3(16,ITERSIZE,1)>>>(cuMat,dim,k);
		cuFloydDDKnl<<<dim3(dim.stride/ITERSIZE,dim.stride/ITERSIZE,1),dim3(16,ITERSIZE,1)>>>(cuMat,dim,k);
	}
}

void cuFloyd_D(double* cuMat,MatrixDim dim)
{
	for (int k=0;k<dim.stride;k+=ITERSIZE)
	{
		cuFloydIDKnl<<<1,dim3(ITERSIZE,16,1)>>>(cuMat,dim,k);
		cuFloydSDKnl<<<dim3(dim.stride/ITERSIZE,2,1),dim3(16,ITERSIZE,1)>>>(cuMat,dim,k);
		cuFloydDDKnl<<<dim3(dim.stride/ITERSIZE,dim.stride/ITERSIZE,1),dim3(16,ITERSIZE,1)>>>(cuMat,dim,k);
	}
}

void cuFill_F(float* cuMat,MatrixDim dim)
{
	cuFillKnl<<<dim3(dim.stride/ITERSIZE,dim.stride,1),dim3(ITERSIZE,1,1)>>>(cuMat,dim,(float)1e20);
}

void cuFill_D(double* cuMat,MatrixDim dim)
{
	cuFillKnl<<<dim3(dim.stride/ITERSIZE,dim.stride,1),dim3(ITERSIZE,1,1)>>>(cuMat,dim,(double)1e20);
}

void cuFill_I(int* cuMat,MatrixDim dim)
{
	cuFillKnl<<<dim3(dim.stride/ITERSIZE,dim.stride,1),dim3(ITERSIZE,1,1)>>>(cuMat,dim,CUMAXINT);
}
