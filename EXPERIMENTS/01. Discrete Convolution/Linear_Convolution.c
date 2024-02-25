#include<stdio.h>

void Linear_Convolution(float *x, int N, float *h, float *y);

int main()
{
    
    int i,j,k, L,M,N;
    float y[20],x[10],h[10];

    
    for(i=0;i<10;i++)
    {
	 x[i]=h[i]=y[i]=0;
    }

    printf( "\n Enter the length of x[n] ie L = ");
    scanf("%d", &L);

    printf( "Enter the values of x[n] \n");
    for(i=0;i<L;i++)
    {
	scanf("%f",&x[i]);
    }

    printf( "\n Enter the length of h[n] ie M = ");
    scanf("%d", &M);

    printf( "Enter the values of h[n] \n");

    for(i=0;i<M;i++)
    {
	scanf("%f",&h[i]);
    }

    N = L + M - 1;

    Linear_Convolution(x,N,h,y);

    printf("\n\n x[n] = " );
    for(i=0;i<N;i++)	printf("  %4.2f  ",x[i]);

    printf("\n\n h[n] = " );
    for(i=0;i<N;i++)    printf("  %4.2f  ",h[i]);

    printf("\n\n y[n] = " );
    for(i=0;i<N;i++)    printf("  %4.2f  ",y[i]);
    
    printf("\n\n" );
    
    

}


void Linear_Convolution(float *x, int N, float *h, float *y)
{


    int i,j,k;
    float temp=0;

    for(k=0;k<N;k++)
    {
	i=k+1;
	temp=0;
	for(j=0;j<N;j++)
	{
		i--;
		temp += x[j] * h[i];

		if(i==0) i += N;
	}
	y[k] = temp;
    }

}