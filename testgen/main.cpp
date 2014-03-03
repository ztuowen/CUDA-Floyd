#include<iostream>
#include<fstream>
#include<stdio.h>
#include<time.h>
#include<cstdlib>

using namespace std;

int main(int argc,char **args)
{
	if (argc<2)
	{
		cout<<"Usage: washall-test [N] (filename)"<<endl;
		cout<<"[N] is the number of nodes in the graph"<<endl;
		cout<<"Output: Default: inmatrix.txt or filename specified by the user"<<endl;
		exit(1);
	}
	ofstream co;
	if (argc>2)
		co.open(args[2]);
	else
		co.open("inmatrix.txt");
	int N;
	srand((unsigned)time(NULL));
	sscanf(args[1],"%d",&N);
	co<<N<<endl;
	for (int i=0;i<N;++i)
	{
		for (int j=0;j<N;++j)
			if (i!=j)
				//co<<rand()/(float)RAND_MAX<<"\t";
				co<<rand()%65536<<"\t";
			else
				co<<"0\t";
		co<<endl;
	}
	cout<<"Test file generated"<<endl;
	return 0;
}
