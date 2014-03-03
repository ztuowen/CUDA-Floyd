/*
 * main.cu
 *
 *  Created on: Jan 11, 2014
 *      Author: joe
 */

#include <stdio.h>
#include <stdlib.h>
#include "adjmatrix.h"
#include <fstream>
#include <iostream>

using namespace std;

int main(int argc,char **args) {

	ifstream ci;
	ofstream co;

	AdjMatrix<float> mat;

	if (argc>2)
	{
		co.open(args[1]);
		ci.open(args[2]);
	}
	else
	{
		co.open("outmatrix.txt");
		ci.open("inmatrix.txt");
	}

	mat.input(ci);

	mat.floyd();

	mat.output(co);

	co.close();
	ci.close();

	//CUDA_CHECK_RETURN(cudaDeviceReset());

	return 0;
}
