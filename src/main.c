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
#define DELHI
//-------------------------------------------------------

//-------------------------------------------------------
//SELECT POLYNOMIAL DEGREE
#define POLY_DEGREE 1
//-------------------------------------------------------


#ifdef DELHI
	#include "../data/delhi.h"
	#include "def_delhi.h"
#endif

#ifdef TURBINE
	#include "../data/turbine.h"
	#include "def_turbine.h"
#endif



float polynomial_regression_train_and_test_optimized(float data[SAMPLE_SIZE][DATASET_FEATURES], int feature_size, int sample_size, int degree);
float polynomial_regression_train_and_test(float data[SAMPLE_SIZE][DATASET_FEATURES], int feature_size, int sample_size, int degree);


int main(void)
{
	uint32_t start_time, end_time;
	float rmse;

	//Enable clock cycle register and reset, then store the start of clock cycles.
	DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
	DWT->CYCCNT = 0;

	start_time = DWT->CYCCNT;

    //pick one
	rmse = polynomial_regression_train_and_test(DATASET, DATASET_FEATURES, SAMPLE_SIZE, POLY_DEGREE);

	end_time = DWT->CYCCNT;

	//calculate elapsed clock cycles
	uint32_t elapsed_clocks = end_time - start_time;


	//"useless" statements, used to analyze memory during debugging
	elapsed_clocks;
	rmse;

	while(1);
}

float polynomial_regression_train_and_test_optimized(float data[SAMPLE_SIZE][DATASET_FEATURES], int feature_size, int sample_size, int degree)
{
    // compute number of terms in polynomial regression
    int NUM_TERMS = 1;
    for(int i = 0; i < feature_size - 1; i++)
	{
		NUM_TERMS *= (degree + 1);
	}
        
    // allocate memory for feature matrix (x) and label array (y)
    float **x = malloc(sample_size * sizeof(float *));
    float *y = malloc(sample_size * sizeof(float));

    // feature matrix and label array
    for(int i = 0; i < sample_size; i++)
    {
        x[i] = malloc(NUM_TERMS * sizeof(float));
        int index = 0;

        for(int comb = 0; comb < NUM_TERMS; comb++)
        {
            int temp_comb = comb;
            float value = 1;

            // calculate product of features raised to powers
            for(int feature = 0; feature < feature_size - 1; feature++)
            {
                int power = temp_comb % (degree + 1);
                float temp = 1;

                //OPTIMIZED WITH MULTIPLICATION LOOP
                for(int p = 0; p < power; p++)
                {
                    temp *= data[i][feature];
                }

                value *= temp;
                temp_comb /= (degree + 1);
            }

            x[i][index] = value;
            index++;
        }

        // label value
        y[i] = data[i][feature_size - 1];
    }

    // allocate memory for x transpose, x transpose times x, and x transpose times y matrices
    float **x_transpose = malloc(NUM_TERMS * sizeof(float *));
    float **x_transpose_x = malloc(NUM_TERMS * sizeof(float *));
    float *x_transpose_y = malloc(NUM_TERMS * sizeof(float));

    for(int i = 0; i < NUM_TERMS; i++)
    {
        x_transpose[i] = malloc(sample_size * sizeof(float));
        x_transpose_x[i] = malloc(NUM_TERMS * sizeof(float));
        float sum_y = 0;

        for(int j = 0; j < sample_size; j++)
        {
            x_transpose[i][j] = x[j][i];
            sum_y += x_transpose[i][j] * y[j];
        }

        x_transpose_y[i] = sum_y;

        for(int j = 0; j < NUM_TERMS; j++)
        {
            float sum_x = 0;

            for(int k = 0; k < sample_size; k++)
            {
                sum_x += x_transpose[i][k] * x[k][j];
            }

            x_transpose_x[i][j] = sum_x;
        }
    }

    // allocate memory for coefficients
    float *coefficients = malloc(NUM_TERMS * sizeof(float));

    // gaussian elimination
    for(int i = 0; i < NUM_TERMS; i++)
    {
        for(int j = i + 1; j < NUM_TERMS; j++)
        {
            float ratio = x_transpose_x[j][i] / x_transpose_x[i][i];

            for(int k = i; k < NUM_TERMS; k++)
            {
                x_transpose_x[j][k] -= ratio * x_transpose_x[i][k];
            }

            x_transpose_y[j] -= ratio * x_transpose_y[i];
        }
    }

    // back-substitution to find coefficients
    for(int i = NUM_TERMS - 1; i >= 0; i--)
    {
        float sum = 0;

        for(int j = i + 1; j < NUM_TERMS; j++)
        {
            sum += x_transpose_x[i][j] * coefficients[j];
        }

        coefficients[i] = (x_transpose_y[i] - sum) / x_transpose_x[i][i];
    }

    // calculate RMSE on training data
    float rmse = 0;

    for(int i = 0; i < sample_size; i++)
    {
        float y_pred = 0;

        for(int j = 0; j < NUM_TERMS; j++)
        {
            y_pred += coefficients[j] * x[i][j];
        }

        rmse += (y_pred - data[i][feature_size - 1]) * (y_pred - data[i][feature_size - 1]);
    }

    rmse = sqrt(rmse / sample_size);

    // free all memory
    for (int i = 0; i < sample_size; i++)
    {
        free(x[i]);
    }

    free(x);
    free(y);

    for (int i = 0; i < NUM_TERMS; i++)
    {
        free(x_transpose[i]);
        free(x_transpose_x[i]);
    }

    free(x_transpose);
    free(x_transpose_x);
    free(x_transpose_y);
    free(coefficients);

    return rmse;
}

float polynomial_regression_train_and_test(float data[SAMPLE_SIZE][DATASET_FEATURES], int feature_size, int sample_size, int degree)
{
    // compute number of terms in polynomial regression
    int NUM_TERMS = (int)pow(degree + 1, feature_size - 1);

    // allocate memory for feature matrix (x) and label array (y)
    float **x = malloc(sample_size * sizeof(float *));
    float *y = malloc(sample_size * sizeof(float));

    // populate x and y from input data
    for(int i = 0; i < sample_size; i++)
    {
        x[i] = malloc(NUM_TERMS * sizeof(float));
        int index = 0;

        // populate feature matrix (x)
        for(int comb = 0; comb < NUM_TERMS; comb++)
        {
            int temp_comb = comb;
            float value = 1;

            // calculate product of features raised to powers
            for(int feature = 0; feature < feature_size - 1; feature++)
            {
                int power = temp_comb % (degree + 1);
                value *= pow(data[i][feature], power);
                temp_comb /= (degree + 1);
            }

            x[i][index] = value;
            index++;
        }

        // store label value (y)
        y[i] = data[i][feature_size - 1];
    }

    // allocate memory for x transpose, x transpose times x, and x transpose times y matrices
    float **x_transpose = malloc(NUM_TERMS * sizeof(float *));
    float **x_transpose_x = malloc(NUM_TERMS * sizeof(float *));
    float *x_transpose_y = malloc(NUM_TERMS * sizeof(float));

    // compute x transpose and x transpose times x matrices
    for(int i = 0; i < NUM_TERMS; i++)
    {
        x_transpose[i] = malloc(sample_size * sizeof(float));
        x_transpose_x[i] = malloc(NUM_TERMS * sizeof(float));

        // Compute x transpose matrix
        for(int j = 0; j < sample_size; j++)
        {
            x_transpose[i][j] = x[j][i];
        }

        // Compute x transpose times x matrix
        for(int j = 0; j < NUM_TERMS; j++)
        {
            x_transpose_x[i][j] = 0;

            for(int k = 0; k < sample_size; k++)
            {
                x_transpose_x[i][j] += x_transpose[i][k] * x[k][j];
            }
        }

        // Compute x transpose times y matrix
        x_transpose_y[i] = 0;

        for(int j = 0; j < sample_size; j++)
        {
            x_transpose_y[i] += x_transpose[i][j] * y[j];
        }
    }

    // allocate memory for coefficients
    float *coefficients = malloc(NUM_TERMS * sizeof(float));

    // gaussian elimination to solve system of equations
    for(int i = 0; i < NUM_TERMS; i++)
    {
        for(int j = i + 1; j < NUM_TERMS; j++)
        {
            float ratio = x_transpose_x[j][i] / x_transpose_x[i][i];

            for(int k = i; k < NUM_TERMS; k++)
            {
                x_transpose_x[j][k] -= ratio * x_transpose_x[i][k];
            }

            x_transpose_y[j] -= ratio * x_transpose_y[i];
		}

    }

    // back-substitution to find coefficients
    for(int i = NUM_TERMS - 1; i >= 0; i--)
    {
        float sum = 0;

        for(int j = i + 1; j < NUM_TERMS; j++)
        {
            sum += x_transpose_x[i][j] * coefficients[j];
        }

        coefficients[i] = (x_transpose_y[i] - sum) / x_transpose_x[i][i];
    }

    // calculate RMSE on training data
    float rmse = 0;

    for(int i = 0; i < sample_size; i++)
    {
        float y_pred = 0;

        for(int j = 0; j < NUM_TERMS; j++)
        {
            y_pred += coefficients[j] * x[i][j];
        }

        rmse += pow(y_pred - data[i][feature_size - 1], 2);
    }

    rmse = sqrt(rmse / sample_size);

    // free all memory
    for(int i = 0; i < sample_size; i++)
    {
        free(x[i]);
    }

    free(x);
    free(y);

    for(int i = 0; i < NUM_TERMS; i++)
    {
        free(x_transpose[i]);
        free(x_transpose_x[i]);
    }

    free(x_transpose);
    free(x_transpose_x);
    free(x_transpose_y);
    free(coefficients);

    return rmse;
}