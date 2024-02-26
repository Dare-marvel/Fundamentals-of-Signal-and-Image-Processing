#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define max 4

void DITFFT_4_Point(int N, float x[4][2], float t[4][2]);
void DITFFT_8_Point(int N, float x[8][2], float t[8][2]);

void main()
{

	int i, j, k, n, L, N;
	float x[8][2], X[8][2], t[8][2];

	// Real & Imaginary part initialization

	for (i = 0; i < max; i++) // initilization
	{
		X[i][0] = 0;
		X[i][1] = 0; // Input X[k] == 0
		x[i][0] = 0;
		x[i][1] = 0; // Output x[n] = 0
	}

	printf("\n\n Enter the length of x[n] ( 4 pt or 8 pt)  = : ");
	scanf("%d", &L);

	// N must be Radix 2 Number for FFT algorithm
	if (L > 4)
		N = 8;
	else
		N = 4;

	printf(" Enter the values of x[n]     : ");

	for (i = 0; i < L; i++)
	{
		scanf("%f", &x[i][0]);
	}

	printf("\n\n Input signal x[n] = ");
	for (i = 0; i < L; i++)
		printf("  %4.2f  ", x[i][0]);

	// ----  DIT-FFT---------

	if (N == 4)
		DITFFT_4_Point(N, x, X);
	else if (N == 8)
		DITFFT_8_Point(N, x, X);
	else
		exit(0);

	printf("\n\n FFT results X[k] =  :\n  ");

	for (k = 0; k < N; k++)
		printf("\n %7.3f  + j  %7.3f", X[k][0], X[k][1]);

	printf("\n\n");

	// ----  Inverse DIT-FFT---------

	for (i = 0; i < N; i++) // initilization
	{
		x[i][0] = 0;
		x[i][1] = 0; // Output x[n] = 0
	}

	// (I) Find X*[k]

	for (k = 0; k < N; k++)
	{
		X[k][1] = X[k][1] * (-1);
	}

	// (II) Find FFT{X*[k]}

	if (N == 4)
		DITFFT_4_Point(N, X, x);
	else if (N == 8)
		DITFFT_8_Point(N, X, x);
	else
		exit(0);

	// (III) Find x[n] = {  FFT{X*[k]} / N  }*

	for (n = 0; n < N; n++)
	{
		x[n][0] = (float)x[n][0] / N;
		x[n][1] = (-1) * x[n][1] / N;
	}

	printf("\n Inverse FFT results x[n] =  :\n  ");

	for (n = 0; n < N; n++)
		printf("\n %7.3f  + j  %7.3f", x[n][0], x[n][1]);

	printf("\n\n");
}

void DITFFT_8_Point(int N, float x[8][2], float t[8][2])
{

	int i, j, k, a, b, c, d;
	float e;
	float X1[4][2], X2[4][2], G[4][2], H[4][2];

	for (i = 0; i < 4; i++)
	{
		X1[i][0] = x[2 * i][0];
		X1[i][1] = x[2 * i][1];

		X2[i][0] = x[2 * i + 1][0];
		X2[i][1] = x[(2 * i) + 1][1];
	}

	DITFFT_4_Point(4, X1, G);

	DITFFT_4_Point(4, X2, H);

	// X[k] = G[k] + W H[k]

	e = 6.283185307179586 / N;

	for (k = 0; k < 4; k++)
	{

		t[k][0] = G[k][0] + (H[k][0] * cos(e * k) + H[k][1] * sin(e * k));
		t[k][1] = G[k][1] + (H[k][1] * cos(e * k) - H[k][0] * sin(e * k));
	}

	for (k = 0; k < 4; k++)
	{
		d = k + 4;
		t[d][0] = G[k][0] + (H[k][0] * cos(e * d) + H[k][1] * sin(e * d));
		t[d][1] = G[k][1] + (H[k][1] * cos(e * d) - H[k][0] * sin(e * d));
	}
}

void DITFFT_4_Point(int N, float x[4][2], float t[4][2])
{

	int i, j, k, n, a, b, c, d;
	float e;
	float G[4][2], H[4][2];

	for (n = 0; n < N; n++)
	{
		t[n][0] = 0;
		t[n][1] = 0;
		G[n][0] = 0;
		G[n][1] = 0;
		H[n][0] = 0;
		H[n][1] = 0;
	}

	// Stage-1

	G[0][0] = x[0][0] + x[2][0];
	G[0][1] = x[0][1] + x[2][1];

	G[1][0] = x[0][0] - x[2][0];
	G[1][1] = x[0][1] - x[2][1];

	H[0][0] = x[1][0] + x[3][0];
	H[0][1] = x[1][1] + x[3][1];

	H[1][0] = x[1][0] - x[3][0];
	H[1][1] = x[1][1] - x[3][1];

	// Stage-2

	e = 6.283185307179586 / N;

	// X[k] = G[k] + WNnk H[k]

	k = 0;
	t[0][0] = G[0][0] + (H[0][0] * cos(e * k) + H[0][1] * sin(e * k));
	t[0][1] = G[0][1] + (H[0][1] * cos(e * k) - H[0][0] * sin(e * k));

	k = 1;
	t[1][0] = G[1][0] + (H[1][0] * cos(e * k) + H[1][1] * sin(e * k));
	t[1][1] = G[1][1] + (H[1][1] * cos(e * k) - H[1][0] * sin(e * k));

	k = 2;
	t[2][0] = G[0][0] + (H[0][0] * cos(e * k) + H[0][1] * sin(e * k));
	t[2][1] = G[0][1] + (H[0][1] * cos(e * k) - H[0][0] * sin(e * k));

	k = 3;
	t[3][0] = G[1][0] + (H[1][0] * cos(e * k) + H[1][1] * sin(e * k));
	t[3][1] = G[1][1] + (H[1][1] * cos(e * k) - H[1][0] * sin(e * k));
}