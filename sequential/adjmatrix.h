#ifndef ADJMATRIX_H_
#define ADJMATRIX_H_

#include<iostream>
#include<stdio.h>
#include<cstring>

typedef struct
{
	unsigned int num;
	//unsigned int stride;
} MatrixDim;

template<typename T>
class AdjMatrix
{
private:
	bool init;
	MatrixDim dim;
	T* Mat;
	void input_i(std::istream &ci)
	{
		MatrixDim d;
		ci>>d.num;
		T *ptr=new T[d.num*d.num];
		for (int i=0;i<d.num;++i)
			for (int j=0;j<d.num;++j)
				ci>>ptr[i*d.num+j];
		input_i(d,ptr);
	}
	void input_i(MatrixDim &idim, T* imat)
	{
		dim=idim;
		Mat=imat;
		init=true;
	}
public:
	AdjMatrix()
	{
		init=false;
		Mat=NULL;
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
			delete[] Mat;
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
		co<<dim.num<<std::endl;
		for (int x=0;x<dim.num;++x)
		{
			for (int y=0;y<dim.num;++y)
				co<<Mat[x*dim.num+y]<<"\t";
			co<<std::endl;
		}
	}
	T* getMatrix()
	{
		T *ptr=new T[dim.num*dim.num];
		memcpy(ptr,Mat,sizeof(T)*dim.num*dim.num);
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
	void floyd()
	{
		T tmp,old;
		for (int k=0;k<dim.num;++k)
			for (int i=0;i<dim.num;++i)
				for (int j=0;j<dim.num;++j)
				{
					tmp=Mat[i*dim.num+k]+Mat[k*dim.num+j];
					old=Mat[i*dim.num+j];
					Mat[i*dim.num+j]=tmp<old?tmp:old;
				}
	};
};

#endif
