// gpurisk.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

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
   int global_idx = get_global_id(0);

   int i;

	for (i = 0; i < test_x; i++) 
	{
	   output->totalsNumbers[ global_idx ] += input->inputNumbers[global_idx][i];
	}
}

)V0G0N";

	openClProgram<testInputStruct, testOutputStruct> program(vogon_poem);

	program.RunKernel("test_numbers", &in, &out, 5, 1);


	for (int i = 0; i < test_y; i++)
	{
		std::cout << out.totalsNumbers[i] << std::endl;
	}

}

void riskOpenClTest()
{
	io::file_data fd("gslbeta.cl");

	openClProgram<testInputStruct, testOutputStruct> program(fd.get_data());

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

