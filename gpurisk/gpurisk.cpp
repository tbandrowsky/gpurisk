// gpurisk.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "gslport.h"
#include <iomanip>

const int test_x = 10;
const int test_y = 10;

struct testInputStruct
{
	double inputNumbers[test_y][test_x];
} ;

struct testOutputStruct
{
	double totalsNumbers[test_y];
} ;

struct beta_request
{
	double x, a, b;
};

typedef struct beta_request beta_request;

struct beta_response
{
	int threadid;
	double result;
};

typedef struct beta_response beta_response;

void simpleOpenCLTest()
{
	testInputStruct in;
	testOutputStruct out;

	for (int i = 0; i < test_y; i++)
	{
		for (int j = 0; j < test_x; j++)
		{
			in.inputNumbers[i][j] = i+1;
		}
		out.totalsNumbers[i] = 0;
	}

	const char * vogon_poem = R"V0G0N(

	#define test_x 10
	#define test_y 10

	struct testInputStruct
	{
		double inputNumbers[test_y][test_x];
	};

	typedef struct testInputStruct testInputStruct;

	struct testOutputStruct
	{
		double totalsNumbers[test_y];
	};

	typedef struct testOutputStruct testOutputStruct;

__kernel void test_numbers(__global testInputStruct* input, __global testOutputStruct* output) 
{
   int global_idx = get_global_id(0);

   int i;

	for (i = 0; i < test_x; i++) 
	{
	   output->totalsNumbers[ global_idx ] += input->inputNumbers[global_idx][i];
	}
}

)V0G0N";

	openClProgram<testInputStruct, testOutputStruct> program(vogon_poem);

	program.RunKernel("test_numbers", &in, &out, 1, 10);

	for (int i = 0; i < test_y; i++)
	{
		std::cout << out.totalsNumbers[i] << std::endl;
	}

}

void riskOpenClTest()
{
	io::file_data fdGsl("gslbeta.cl"),
				  fdNative("nativebeta.cl");

	const int num_requests = 10000000;
	const int group_size = num_requests / 10;

	std::unique_ptr<beta_request[]> requests( new beta_request[num_requests] );

	std::unique_ptr<beta_response[]>	
			responses_gpu(new beta_response[num_requests]),
			responses_cpu(new beta_response[num_requests]),
			responses_stock(new beta_response[num_requests]);

	for (int i = 0; i < num_requests; i++)
	{
		auto req = &requests[i];
		switch (i / group_size)
		{
		case 0:
			req->a = .5;
			req->b = .5;
			break;
		case 1:
			req->a = 5;
			req->b = 1;
			break;
		case 2:
			req->a = 1;
			req->b = 3;
			break;
		case 3:
			req->a = 2;
			req->b = 2;
			break;
		case 4:
			req->a = 2;
			req->b = 5;
			break;
		case 5:
			req->a = .1;
			req->b = .1;
			break;
		case 6:
			req->a = 0.01;
			req->b = 10;
			break;
		case 7:
			req->a = 10;
			req->b = 0.01;
			break;
		case 8:
			req->a = 100;
			req->b = 1;
			break;
		case 9:
			req->a = 1;
			req->b = 100;
			break;
		}
		req->x = (double)(i % group_size) / (double)group_size;
		responses_gpu[i].result = responses_cpu[i].result = responses_stock[i].result = -1.0;
	}

	std::cout << "Running Stock GSL" << std::endl;

	{
		sys::benchmarker bmStock;
		bmStock.start();
		for (int i = 0; i < num_requests; i++)
		{
			responses_stock[i].result = gsl::gsl_cdf_beta_Q(requests[i].x, requests[i].a, requests[i].b);
		}
		bmStock.stop();

		std::cout << "Ran stock " << num_requests << " beta Q's in " << bmStock.getTotalSeconds() << " seconds" << std::endl;
	}

	std::cout << "Running GPU Native" << std::endl;

	{
		sys::benchmarker bmGPU;
		openClProgram<beta_request, beta_response> programGpu(fdNative.get_data(), CL_DEVICE_TYPE_GPU);

		bmGPU.start();
		programGpu.RunKernel("incBetaQ", requests.get(), responses_gpu.get(), num_requests, 1);
		bmGPU.stop();

		std::cout << "Ran GPU " << num_requests << " beta Q's in " << bmGPU.getTotalSeconds() << " seconds" << std::endl;
	}

	std::cout << "Running CPU" << std::endl;
	{

		sys::benchmarker bmCPU;
		openClProgram<beta_request, beta_response> programCpu(fdNative.get_data(), CL_DEVICE_TYPE_CPU);

		bmCPU.start();
		programCpu.RunKernel("incBetaQ", requests.get(), responses_cpu.get(), num_requests, 1);
		bmCPU.stop();

		std::cout << "Ran CPU " << num_requests << " beta Q's in " << bmCPU.getTotalSeconds() << " seconds" << std::endl;
	}

	int cw = 15;
	std::cout << "Differences\n";
	std::cout << std::setw(cw) << "x" << std::setw(cw) << "a" << std::setw(cw) << "b" << std::setw(cw) << "cpu" << std::setw(cw) << "gpu" << std::setw(cw) << "gsl" << std::setw(cw) << "cpu thr" << std::setw(cw) << "gpu thr" << std::endl;
	for (int i = 0; i < num_requests; i++)
	{
		if (fabs(responses_gpu[i].result - responses_stock[i].result) > 0.000001 || num_requests < 101) {
			std::cout << std::setw(cw) 
				<< requests[i].x 
				<< std::setw(cw)
				<< requests[i].a 
				<< std::setw(cw)
				<< requests[i].b
				<< std::setw(cw)
				<< responses_cpu[i].result
				<< std::setw(cw)
				<< responses_gpu[i].result
				<< std::setw(cw)
				<< responses_stock[i].result
				<< std::setw(cw)
				<< responses_cpu[i].threadid
				<< std::setw(cw)
			<< responses_gpu[i].threadid
			<< std::endl;
		}
	}
}

int main()
{
	try
	{
		riskOpenClTest();
		//simpleOpenCLTest();
	}
	catch (std::exception& exc)
	{
		std::cout << exc.what() << std::endl;
	}
}

