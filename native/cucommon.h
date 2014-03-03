/*
 * cucommon.h
 *
 *  Created on: Jan 11, 2014
 *      Author: joe
 */

#ifndef CUCOMMON_H_
#define CUCOMMON_H_

#include<cuda_runtime_api.h>
#include<driver_types.h>
#include<cstdlib>

#define BLKSIZE (256)
#define ITERSIZE (64)
#define TRDPMP	2048
#define LEN (8)
#define SHRINKSZ 16
#define SHRINKSH 4
#define CUMAXINT 67108864

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }


#endif /* CUCOMMON_H_ */
