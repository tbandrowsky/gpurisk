
#include "stdafx.h"

namespace sys
{
	__int64 benchmarker::ticksPerSecond = 0I64;

	__int64 getCpuTicks()
	{
		__int64 ticks;
		QueryPerformanceCounter((LARGE_INTEGER *)&ticks);
		return ticks;
	}

	benchmarker::benchmarker()
	{
		total = 0;
		pass = 0;
		count = 0;

		if (!ticksPerSecond) {
			calibrate();
		}
	}

	void benchmarker::start()
	{
		count++;
		pass = getCpuTicks();
	}

	void benchmarker::stop()
	{
		__int64 stop = getCpuTicks();
		__int64 timer = stop - pass;
		total += timer;
	}

	void benchmarker::reset()
	{
		total = pass = count = 0;
	}

	benchmarker::benchmarker(__int64 _total)
	{
		total = _total;
		pass = 0;
		count = 0;
	}

	benchmarker::~benchmarker()
	{
		;
	}

	void benchmarker::calibrate()
	{
		QueryPerformanceFrequency((LARGE_INTEGER *)(&ticksPerSecond));
	}

	double benchmarker::operator / (const benchmarker& _src)
	{
		return (double)(total) / (double)(_src.getTotal()) * 100.0; // calcs a percentage
	}

	benchmarker benchmarker::operator + (const benchmarker& _src)
	{
		return benchmarker(total + _src.total);
	}

}