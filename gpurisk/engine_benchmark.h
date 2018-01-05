#pragma once

namespace sys
{

	__int64 get_cpu_ticks();

	class benchmarker {

		__int64
			count,
			total,
			pass;

		static __int64 ticksPerSecond;
		static void calibrate();

	public:

		inline __int64 getTotal() const { return total; }
		inline __int64 getPass() const { return pass; }
		static __int64 getTicksPerSecond() { if (!ticksPerSecond) calibrate(); return ticksPerSecond; }

		double getTotalSeconds() const { return (double)(total) / (double)(ticksPerSecond); }
		double getTotalMilliseconds() const { return (double)(total * 1000.0) / (double)(ticksPerSecond); }
		__int64 getTotalTicks() const { return total; }
		double getAvgMilliseconds() const { return count ? getTotalMilliseconds() / (double)count : 0.0; }

		benchmarker(__int64 total);
		benchmarker();
		~benchmarker();

		void start();
		void stop();
		void reset();

		double operator / (const benchmarker& _src);
		benchmarker operator + (const benchmarker& _src);

	};

}
