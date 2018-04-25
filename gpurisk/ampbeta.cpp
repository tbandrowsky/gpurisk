
#include "stdafx.h"

/*
* zlib License
*
* Regularized Incomplete Beta Function
*
* Copyright (c) 2016, 2017 Lewis Van Winkle
* http://CodePlea.com
*
* This software is provided 'as-is', without any express or implied
* warranty. In no event will the authors be held liable for any damages
* arising from the use of this software.
*
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
*
* 1. The origin of this software must not be misrepresented; you must not
*    claim that you wrote the original software. If you use this software
*    in a product, an acknowledgement in the product documentation would be
*    appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
*    misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
*
*  This  port by Todd Bandrowsky
*/

#define STOP 1.0e-8
#define TINY 1.0e-30
#define ERR_VALUE 0;

double incbetaimpl(double x, double a, double b) restrict (amp)
{
	using namespace concurrency::precise_math;

	bool invert = false;
	if (x < 0.0 || x > 1.0) return ERR_VALUE;

	/*The continued fraction converges nicely for x < (a+1)/(a+b+2), SO Use the fact that beta is symmetrical.*/
	if (x > (a + 1.0) / (a + b + 2.0)) {
		double t = a;
		a = b;
		b = t;
		x = 1.0 - x;
		invert = true;
	}

	int sign;
	/*Find the first part before the continued fraction.*/
	const double lbeta_ab = lgamma(a,&sign) + lgamma(b, &sign) - lgamma(a + b, &sign);
	const double front = exp(log(x)*a + log(1.0 - x)*b - lbeta_ab) / a;

	/*Use Lentz's algorithm to evaluate the continued fraction.*/
	double f = 1.0, c = 1.0, d = 0.0;

	int i, m;
	for (i = 0; i <= 200; ++i) {
		m = i / 2;

		double numerator;
		if (i == 0) {
			numerator = 1.0; /*First numerator is 1.0.*/
		}
		else if (i % 2 == 0) {
			numerator = (m*(b - m)*x) / ((a + 2.0*m - 1.0)*(a + 2.0*m)); /*Even term.*/
		}
		else {
			numerator = -((a + m)*(a + b + m)*x) / ((a + 2.0*m)*(a + 2.0*m + 1)); /*Odd term.*/
		}

		/*Do an iteration of Lentz's algorithm.*/
		d = 1.0 + numerator * d;
		if (fabs(d) < TINY) d = TINY;
		d = 1.0 / d;

		c = 1.0 + numerator / c;
		if (fabs(c) < TINY) c = TINY;

		const double cd = c*d;
		f *= cd;

		/*Check for stop.*/
		if (fabs(1.0 - cd) < STOP) {
			double v = front * (f - 1.0);
			if (invert) {
				v = 1.0 - v;
			}
			return v;
		}
	}

	return ERR_VALUE; /*Needed more loops, did not converge.*/
}

void incBeta(concurrency::array_view<beta_request,1> request, concurrency::array_view<beta_response,1> response)
{
	parallel_for_each(
		// Define the compute domain, which is the set of threads that are created.  
		response.extent,
		// Define the code to run on each thread on the accelerator.  
		[=](concurrency::index<1> idx) restrict(amp)
	{
		response[idx].threadid = idx[0];
		response[idx].result = incbetaimpl(request[idx].x, request[idx].a, request[idx].b);
	});
}

void incBetaQ(concurrency::array_view<beta_request, 1> request, concurrency::array_view<beta_response, 1> response)
{
	parallel_for_each(
		// Define the compute domain, which is the set of threads that are created.  
		response.extent,
		// Define the code to run on each thread on the accelerator.  
		[=](concurrency::index<1> idx) restrict(amp)
	{
		response[idx].threadid = idx[0];
		response[idx].result = 1.0 - incbetaimpl(request[idx].x, request[idx].a, request[idx].b);
	});
}

