#pragma once

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

void incBeta(concurrency::array_view<beta_request, 1> request, concurrency::array_view<beta_response, 1> response);
void incBetaQ(concurrency::array_view<beta_request, 1> request, concurrency::array_view<beta_response, 1> response);
