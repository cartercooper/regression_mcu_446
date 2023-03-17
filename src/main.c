/**
  ******************************************************************************
  * @file    main.c
  * @author  Carter Cooper
  * @version V1.0
  * @date    27-January-2023
  * @brief   Default main function.
  ******************************************************************************
*/


#include "stm32f4xx.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include "../data/dehli_raw.h"
#include "../data/turbine_raw.h"

#include "def.h"
			

int main(void)
{
	struct timeval start, end;
  long mtime, secs, usecs; 
  double rmse;

  gettimeofday(&start, NULL);

  rmse = polynomial_regression_train_and_test(DATASET, DATASET_FEATURES, SAMPLE_SIZE, POLY_DEGREE);

  gettimeofday(&end, NULL);
  secs  = end.tv_sec  - start.tv_sec;
  usecs = end.tv_usec - start.tv_usec;
  mtime = ((secs) * 1000000 + usecs) + 0.5;

  printf("POLYNOMIAL REGRESSION: DEGREE %d\n", POLY_DEGREE);
  printf("TIME ELAPSED: %lu\n", mtime);
  printf("RMSE: %.2f\n", rmse);

  while(1);
}

double polynomial_regression_train_and_test(double data[SAMPLE_SIZE][DATASET_FEATURES], int feature_size, int sample_size, int degree)
{
  // Extract features and labels from input matrix
  double **x = malloc(sample_size * sizeof(double *));
  double *y = malloc(sample_size * sizeof(double));
  for (int i = 0; i < sample_size; i++) {
      x[i] = malloc((degree + 1) * sizeof(double));
      for (int j = 0; j <= degree; j++) {
          x[i][j] = pow(data[i][0], j);
      }
      y[i] = data[i][feature_size - 1];
  }

  // Compute x transpose matrix and x transpose times x matrix
  double **x_transpose = malloc((degree + 1) * sizeof(double *));
  double **x_transpose_x = malloc((degree + 1) * sizeof(double *));
  double *x_transpose_y = malloc((degree + 1) * sizeof(double));
  double *temp = malloc((degree + 1) * sizeof(double));
  for (int i = 0; i <= degree; i++) {
      x_transpose[i] = malloc(sample_size * sizeof(double));
      x_transpose_x[i] = malloc((degree + 1) * sizeof(double));
      for (int j = 0; j < sample_size; j++) {
          x_transpose[i][j] = x[j][i];
      }
      for (int j = 0; j <= degree; j++) {
          x_transpose_x[i][j] = 0;
          for (int k = 0; k < sample_size; k++) {
              x_transpose_x[i][j] += x_transpose[i][k] * x[k][j];
          }
      }
      x_transpose_y[i] = 0;
      for (int j = 0; j < sample_size; j++) {
          x_transpose_y[i] += x_transpose[i][j] * y[j];
      }
  }

  // Solve the system of equations to get coefficients
  double *coefficients = malloc((degree + 1) * sizeof(double));
  for (int i = 0; i <= degree; i++) {
      for (int j = i + 1; j <= degree; j++) {
          double ratio = x_transpose_x[i][j] / x_transpose_x[i][i];
          for (int k = i; k <= degree; k++) {
              x_transpose_x[j][k] -= ratio * x_transpose_x[i][k];
          }
          x_transpose_y[j] -= ratio * x_transpose_y[i];
      }
  }
  for (int i = degree; i >= 0; i--) {
      double sum = 0;
      for (int j = i + 1; j <= degree; j++) {
          sum += x_transpose_x[i][j] * coefficients[j];
      }
      coefficients[i] = (x_transpose_y[i] - sum) / x_transpose_x[i][i];
  }

  // Calculate RMSE on training data
  double rmse = 0;
  for (int i = 0; i < sample_size; i++) {
          double y_pred = 0;
          for (int j = 0; j <= degree; j++) {
              y_pred += coefficients[j] * pow(data[i][0], j);
      }
      rmse += pow(y_pred - data[i][feature_size - 1], 2);
  }

  rmse = sqrt(rmse / sample_size);

  // Free memory allocated for arrays
  for (int i = 0; i < sample_size; i++) {
      free(x[i]);
  }
  free(x);
  free(y);
  for (int i = 0; i <= degree; i++) {
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