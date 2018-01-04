#pragma once

#include <map>
#include "file_data.h"

#include <CL/cl.h>

template <class INPUT, class OUTPUT> class openClProgram 
{

	static_assert(std::is_pod<INPUT>::value, "INPUT must be plain old data - no pointers, no dynamic elements, everything public, a struct.");
	static_assert(std::is_pod<OUTPUT>::value, "OUTPUT must be plain old data - no pointers, no dynamic elements, everything public, a struct.");

	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_program program;

public:

	INPUT input;
	OUTPUT output;

	openClProgram(const char *program_buffer, int gpu_type = CL_DEVICE_TYPE_GPU)
	{
		int err;
		/* Identify a platform */
		err = clGetPlatformIDs(1, &platform, NULL);
		if (err < 0) {
			throw std::exception("Couldn't identify a platform");
		}

		/* Access a device */
		err = clGetDeviceIDs(platform, gpu_type, 1, &device, NULL);
		if (err < 0) {
			throw std::exception("Couldn't identify a GPU");
		}

		context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
		if (err < 0) {
			clReleaseDevice(device);
			throw std::exception("Couldn't create a context");
		}

		size_t program_size = strlen(program_buffer);

		program = clCreateProgramWithSource(context, 1, (const char**)&program_buffer, &program_size, &err);
		if (err < 0) {
			clReleaseContext(context);
			clReleaseDevice(device);
			throw std::exception("Couldn't create program");
		}

		err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
		if (err < 0) {

			size_t log_size;
			/* Find size of log and print to std output */
			clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);			
			char *program_log = (char*)malloc(log_size + 1);
			program_log[log_size] = '\0';
			clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
			std::string plog = program_log;
			free(program_log);

			clReleaseProgram(program);
			clReleaseContext(context);
			clReleaseDevice(device);
			throw std::exception(plog.c_str());
		}
	}

	virtual ~openClProgram()
	{
		clReleaseProgram(program);
		clReleaseContext(context);
		clReleaseDevice(device);
	}

	template <class InputStruct, class OutputStruct> bool RunKernel(const char *kernalName, InputStruct *input, OutputStruct *output, size_t input_size = 1, size_t local_size = 1)
	{
		cl_command_queue queue;
		cl_kernel kernel;
		int err;

		/* Create a command queue */
		queue = clCreateCommandQueue(context, device, 0, &err);
		if (err < 0) {
			throw std::exception("Couldn't create a command queue.");
		};

		/* Create a kernel */
		kernel = clCreateKernel(program, kernalName, &err);
		if (err < 0) {
			clReleaseCommandQueue(queue);
			throw std::exception("Couldn't create a kernal.");
		};

		/* Create data buffer */
		auto input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(InputStruct) * input_size, input, &err);
		if (err < 0) {
			clReleaseKernel(kernel);
			clReleaseCommandQueue(queue);
			throw std::exception("Couldn't input buffer.");
		};

		auto output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |CL_MEM_COPY_HOST_PTR, sizeof(OutputStruct) * input_size, output, &err);
		if (err < 0) {
			clReleaseKernel(kernel);
			clReleaseMemObject(input_buffer);
			clReleaseCommandQueue(queue);
			throw std::exception("Couldn't create output buffer.");
		};

		/* Create kernel arguments */
		err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
		err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
		if (err < 0) {
			clReleaseKernel(kernel);
			clReleaseMemObject(output_buffer);
			clReleaseMemObject(input_buffer);
			clReleaseCommandQueue(queue);
			throw std::exception("Couldn't create kernel argument.");
		}

		/* Enqueue kernel */
		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &input_size,
			&local_size, 0, NULL, NULL);
		if (err < 0) {
			clReleaseKernel(kernel);
			clReleaseMemObject(output_buffer);
			clReleaseMemObject(input_buffer);
			clReleaseCommandQueue(queue);
			throw std::exception("Couldn't enqueue kernel.");
		}

		/* Read the kernel's output */
		err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
			sizeof(OutputStruct)* input_size, output, 0, NULL, NULL);
		if (err < 0) {
			clReleaseKernel(kernel);
			clReleaseMemObject(output_buffer);
			clReleaseMemObject(input_buffer);
			clReleaseCommandQueue(queue);
			throw std::exception("Couldn't read buffer.");
		}

		/* Deallocate resources */
		clReleaseKernel(kernel);
		clReleaseMemObject(output_buffer);
		clReleaseMemObject(input_buffer);
		clReleaseCommandQueue(queue);
	}

};
