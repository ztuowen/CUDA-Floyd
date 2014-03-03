#ifndef ADJMATRIX_H_
#define ADJMATRIX_H_

#include<iostream>
#include<stdio.h>
#include"cukernels.h"
#include"cucommon.h"
#include"cutimer.h"

template<typename T>
class AdjMatrix
{
private:
	bool init;
	MatrixDim dim;
	T* cuMat;
	void input_i(std::istream &ci)
	{
		MatrixDim d;
		ci>>d.num;
		d.stride=(d.num&(~(ITERSIZE-1)))+((d.num&(ITERSIZE-1))>0?ITERSIZE:0);
		T *ptr=new T[d.num*d.stride];
		for (int i=0;i<d.num;++i)
			for (int j=0;j<d.num;++j)
				ci>>ptr[i*d.stride+j];
		input_i(d,ptr);
		delete[] ptr;
	}
	void input_i(MatrixDim &idim, T* imat)
	{
		dim=idim;
		CUDA_CHECK_RETURN(cudaMalloc((void**) &cuMat, sizeof(T) * dim.num*dim.stride));
		CUDA_CHECK_RETURN(cudaMemcpy((void *)cuMat, imat, sizeof(T) * dim.num*dim.stride, cudaMemcpyHostToDevice));
		fill();
		init=true;
	}
public:
	AdjMatrix()
	{
		init=false;
		cuMat=NULL;
	}
	AdjMatrix(std::istream &ci)
	{
		input_i(ci);
	}
	AdjMatrix(MatrixDim &idim, T* imat)
	{
		input_i(idim,imat);
	}
	~AdjMatrix()
	{
		clean();
	}
	void clean()
	{
		if (init)
			CUDA_CHECK_RETURN(cudaFree((void*) cuMat));
		init=false;
	}
	void input(std::istream &ci)
	{
		clean();
		input_i(ci);
	}
	void input(MatrixDim &idim, T* imat)
	{
		clean();
		input_i(idim,imat);
	}
	void output(std::ostream &co)
	{
		T *ptr=getMatrix();
		co<<dim.num<<std::endl;
		for (int x=0;x<dim.num;++x)
		{
			for (int y=0;y<dim.num;++y)
				co<<ptr[x*dim.stride+y]<<"\t";
			co<<std::endl;
		}
		delete[] ptr;
	}
	T* getMatrix()
	{
		T *ptr=new T[dim.num*dim.stride];
		CUDA_CHECK_RETURN(cudaMemcpy(ptr, (void *)cuMat, sizeof(T) * dim.num*dim.stride, cudaMemcpyDeviceToHost));
		return ptr;
	}
	MatrixDim getDim()
	{
		return dim;
	}
	bool hasData()
	{
		return init;
	}
	void floyd(){ std::cout<<"Floyd not implemented"<<std::endl; };
	void fill(){ std::cout<<"Fill not implemented"<<std::endl; };
};

#endif
