#include<stdio.h>
#include<math.h>

#define max 64

void Circular(float *x, int N, float *h, float *y);

int main()
{

    int i,j,k,L,M,N;
    float x[max],y[max];
    float h[max];

    
    for(i=0;i<max;i++)
    {
       x[i]=0; h[i]=0; y[i]=0;
    }
  
    
    printf("Enter the length of x[n] L = : ");
    scanf("%d",&L);
    
    
    printf( "Enter the values of x[n] : ");

    for(i=0;i<L;i++)
    {
		scanf("%f",&x[i]);
    }
    
    printf("Enter the length of h[n] M = : ");
    scanf("%d",&M);
    
    printf( "Enter the values of h[n] : ");

    for(i=0;i<M;i++)
    {
	scanf("%f",&h[i]);
    }

	if( L > M )
	 	N = L;
	else 
		N = M;
		
    
    Circular(x, N, h, y);

    
    printf("\n\n x[n] = " );
    for(i=0;i<N;i++)	
    	printf("  %4.2f  ",x[i]);

    printf("\n\n h[n] = " );
    for(i=0;i<N;i++)    
    	printf("  %4.2f  ",h[i]);

    printf("\n\n y[n] = " );
    for(i=0;i<N;i++)    
    	printf("  %4.2f  ",y[i]);
    	
    printf("\n\n");

}

void Circular(float *x, int N, float *h, float *y)
{
     
        int i,j,k,p,q,s,t;
        float sum=0,temp[1024],signal[1024];

  	j=0;
	  for(i=0;i<N;i++)  /* reverse the signal -*/
	  {
	       j = N-1-i;
	       temp[i]=h[j];
  	}

  	for(s=0;s<N;s++)
  	{
		p=0;
		for(t=0;t<N;t++)
		{
	   		if(t<=s)
	   		{
	      			q=t-s+N-1;
	      			signal[t]=temp[q];
	   		}

	   		else
	   		{
	       			signal[t]=temp[p];
	       			p++;
	   		}
		}
		sum=0;
		for(i=0;i<N;i++)
		{
	     		sum += x[i] * signal[i];
		}
		y[s]=sum;
  
  	}

  

}

