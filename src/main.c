/**
  ******************************************************************************
  * @file    main.c
  * @author  Carter Cooper
  * @version V1.0
  * @date    27-January-2023
  * @brief   Polynomial Regression.
  ******************************************************************************
*/

#include "stm32f4xx.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

//-------------------------------------------------------
//SELECT DATA BEFORE BUILD
#define TURBINE
//-------------------------------------------------------

//-------------------------------------------------------
//SELECT POLYNOMIAL DEGREE
#define POLY_DEGREE 3
//-------------------------------------------------------


#ifdef delhi
	#include "../data/delhi.h"
	#include "def_delhi.h"
#endif

#ifdef TURBINE
	#include "../data/turbine.h"
	#include "def_turbine.h"
#endif



float polynomial_regression_train_and_test(float data[SAMPLE_SIZE][DATASET_FEATURES], int feature_size, int sample_size, int degree);


int main(void)
{
	uint32_t start_time, end_time;
	float rmse;

	//Enable clock cycle register and reset, then store the start of clock cycles.
	DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
	DWT->CYCCNT = 0;

	start_time = DWT->CYCCNT;

	rmse = polynomial_regression_train_and_test(DATASET, DATASET_FEATURES, SAMPLE_SIZE, POLY_DEGREE);

	end_time = DWT->CYCCNT;

	//calculate elapsed clock cycles
	uint32_t elapsed_clocks = end_time - start_time;


	//"useless" statements, used to analyze memory during debugging
	elapsed_clocks;
	rmse;

	while(1);
}

float polynomial_regression_train_and_test(float data[SAMPLE_SIZE][DATASET_FEATURES], int feature_size, int sample_size, int degree)
{
	// Extract features and labels from input matrix
	float **x = malloc(sample_size * sizeof(float *));
	float *y = malloc(sample_size * sizeof(float));

	for (int i = 0; i < sample_size; i++)
	{
		x[i] = malloc((degree + 1) * sizeof(float));

		for (int j = 0; j <= degree; j++)
		{
			x[i][j] = pow(data[i][0], j);
		}

		y[i] = data[i][feature_size - 1];
	}

	// Compute x transpose matrix and x transpose times x matrix
	float **x_transpose = malloc((degree + 1) * sizeof(float *));
	float **x_transpose_x = malloc((degree + 1) * sizeof(float *));
	float *x_transpose_y = malloc((degree + 1) * sizeof(float));
	float *temp = malloc((degree + 1) * sizeof(float));

	for (int i = 0; i <= degree; i++)
	{
		x_transpose[i] = malloc(sample_size * sizeof(float));
		x_transpose_x[i] = malloc((degree + 1) * sizeof(float));

		for (int j = 0; j < sample_size; j++)
		{
			x_transpose[i][j] = x[j][i];
		}

		for (int j = 0; j <= degree; j++)
		{
			x_transpose_x[i][j] = 0;

			for (int k = 0; k < sample_size; k++)
			{
			  x_transpose_x[i][j] += x_transpose[i][k] * x[k][j];
			}
		}

		x_transpose_y[i] = 0;

		for (int j = 0; j < sample_size; j++)
		{
			x_transpose_y[i] += x_transpose[i][j] * y[j];
		}
	}

	// Solve the system of equations to get coefficients
	float *coefficients = malloc((degree + 1) * sizeof(float));

	for (int i = 0; i <= degree; i++)
	{
		for (int j = i + 1; j <= degree; j++)
		{
			float ratio = x_transpose_x[i][j] / x_transpose_x[i][i];

			for (int k = i; k <= degree; k++)
			{
				x_transpose_x[j][k] -= ratio * x_transpose_x[i][k];
			}

			x_transpose_y[j] -= ratio * x_transpose_y[i];
		}
	}

	for (int i = degree; i >= 0; i--)
	{
		float sum = 0;

		for (int j = i + 1; j <= degree; j++)
		{
			sum += x_transpose_x[i][j] * coefficients[j];
		}

		coefficients[i] = (x_transpose_y[i] - sum) / x_transpose_x[i][i];
	}

	// Calculate RMSE on training data
	float rmse = 0;

	for (int i = 0; i < sample_size; i++)
	{
		float y_pred = 0;

		for (int j = 0; j <= degree; j++)
		{
			y_pred += coefficients[j] * pow(data[i][0], j);
		}

		rmse += pow(y_pred - data[i][feature_size - 1], 2);
	}

	rmse = sqrt(rmse / sample_size);

	// Free memory allocated for arrays
	for (int i = 0; i < sample_size; i++)
	{
		free(x[i]);
	}

	free(x);
	free(y);

	for (int i = 0; i <= degree; i++)
	{
		free(x_transpose[i]);
		free(x_transpose_x[i]);
	}
	
	free(x_transpose);
	free(x_transpose_x);
	free(x_transpose_y);
	free(temp);
	free(coefficients);

	return rmse;
}
