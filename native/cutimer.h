/*
 * cuTimer.h
 *
 *  Created on: Jan 13, 2014
 *      Author: joe
 */

#ifndef CUTIMER_H_
#define CUTIMER_H_

class cuTimer
{
	cudaEvent_t start, stop;
	float milsec;
public:
	cuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		milsec=0;
	}
	void logstart()
	{
		cudaEventRecord(start);
	}
	void logstop()
	{
		cudaEventRecord(stop);
	}
	void logtime()
	{
		cudaEventSynchronize(stop);
		milsec = 0;
		cudaEventElapsedTime(&milsec, start, stop);
	}
	float gettime()
	{
		return milsec;
	}
};


#endif /* CUTIMER_H_ */
