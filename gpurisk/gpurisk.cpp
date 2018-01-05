// gpurisk.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "gslport.h"

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

struct gsl_cdf_beta_request
{
	double x, a, b;
};

struct gsl_cdf_beta_response
{
	double result;
};


void simpleOpenCLTest()
{
	testInputStruct in;
	testOutputStruct out;

	for (int i = 0; i < test_y; i++)
	{
		for (int j = 0; j < test_x; j++)
		{
			in.inputNumbers[i][j] = i;
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
   int global_idx = get_local_id(0);

   int i;

	for (i = 0; i < test_x; i++) 
	{
	   output->totalsNumbers[ global_idx ] += input->inputNumbers[global_idx][i];
	}
}

)V0G0N";

	openClProgram<testInputStruct, testOutputStruct> program(vogon_poem);

	program.RunKernel("test_numbers", &in, &out, 1, 5);

	for (int i = 0; i < test_y; i++)
	{
		std::cout << out.totalsNumbers[i] << std::endl;
	}

}

void riskOpenClTest()
{
	io::file_data fd("gslbeta.cl");

	const int num_requests = 1000000;
	const int group_size = num_requests / 5;

	std::unique_ptr<gsl_cdf_beta_request[]> requests( new gsl_cdf_beta_request[num_requests] );

	std::unique_ptr<gsl_cdf_beta_response[]>	
			responses_gpu(new gsl_cdf_beta_response[num_requests]),
			responses_cpu(new gsl_cdf_beta_response[num_requests]),
			responses_stock(new gsl_cdf_beta_response[num_requests]);

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
		}
		req->x = (double)(i % group_size) / (double)group_size;
		responses_gpu[i].result = responses_cpu[i].result = responses_stock[i].result = -1.0;
	}

	std::cout << "Running Stock" << std::endl;

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

	std::cout << "Running GPU" << std::endl;

	{
		sys::benchmarker bmGPU;
		openClProgram<gsl_cdf_beta_request, gsl_cdf_beta_response> programGpu(fd.get_data(), CL_DEVICE_TYPE_GPU);

		bmGPU.start();
		programGpu.RunKernel("gsl_cdf_beta_Q_cl", requests.get(), responses_gpu.get(), num_requests, 1);
		bmGPU.stop();

		std::cout << "Ran GPU " << num_requests << " beta Q's in " << bmGPU.getTotalSeconds() << " seconds" << std::endl;
	}

	/*
	std::cout << "Running CPU" << std::endl;
	{
		openClProgram<gsl_cdf_beta_request, gsl_cdf_beta_response> programCpu(fd.get_data(), CL_DEVICE_TYPE_CPU);
		programCpu.RunKernel("gsl_cdf_beta_Q_cl", requests, responses_cpu, num_requests, 1);
	}
	*/

	std::cout << "Differences\nx\t a\t b\t\t gpu\t\t stock\t\t" << std::endl;
	for (int i = 0; i < num_requests; i++)
	{
		if (fabs(responses_gpu[i].result - responses_stock[i].result) > 0.00001) {
			std::cout << requests[i].x << "\t" << requests[i].a << "\t" << requests[i].b << "\t\t";
//			std::cout << responses_cpu[i].result << "\t\t";
			std::cout << responses_gpu[i].result << "\t\t";
			std::cout << responses_stock[i].result << "\t\t";
			std::cout << std::endl;
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

