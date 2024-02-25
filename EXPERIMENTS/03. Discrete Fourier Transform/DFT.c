#include<stdio.h>
#include<math.h>

void dft(int N,float a[16],float b[16],float A[16],float B[16]);
void idft(int N,float A[16],float B[16],float a[16],float b[16]);

void main()
{

	int i,N;

	float a[16],b[16];
	float A[16],B[16];

	for(i=0;i<16;i++) //Initilization
        {
	   a[i]=0;  b[i]=0;
	}

	printf( "\n Enter the length of x[n] ie N = ");
	scanf("%d",&N);

	printf( "Enter the values of x[n] : ");
	for(i=0;i<N;i++)
		scanf("%f",&a[i]);

	
	
	// ----DFT-------
	
		
	dft(N,a,b,A,B);

	printf("\n X[k] by DFT : ");
	for(i=0;i<N;i++) 
	    printf("\n %6.2f +j  %6.2f  ",A[i],B[i]);

	printf("\n\n");
	
	
	
	//----IDFT-------
	
	
	idft(N,A,B,a,b);
        
        
        printf("\n x[n] by IDFT : ");
	for(i=0;i<N;i++)
	{
	    printf("\n %6.2f + j  %6.2f  ",a[i],b[i]);
	}

        printf("\n\n");

	
	
	
	//exit(0);
}

void dft(int N,float a[16],float b[16],float A[16],float B[16])
{

    int i,k,n;
    float arg,c,d,omega;
   
     
     
   omega=(float)6.2831853/N;
      
  
   for(k = 0; k < N; k++) 
   {
       
       A[k]=0; B[k]=0; 
       
       for(n = 0; n < N; n++) 
       {
		   	
	   	arg=omega*k*n;

	   	c=cos(arg);  d=sin(arg);
	
		// (a+jb)(c+jd) = (ac - bd ) + j (bc + ad)
   		
   		A[k] = A[k] + a[n] * c + b[n] * d ;
   		B[k] = B[k] - a[n] * d + b[n] * c ;
       }
    }
}


void idft(int N,float A[16],float B[16],float a[16],float b[16])
{

   int i,k;
   float arg,c,d,omega;
   
     
     
   omega=(float)6.2831853/N;
      
   for(k=0; k<N; k++)
   {
       a[k]=0;  b[k]=0;
       
              
       for(i=0;i<N;i++)
       {

	   arg = omega * k * i;
	  
	   c = cos(arg);   d = sin(arg);
	   
	   // (a+jb)(c+jd) = (ac - bd ) + j (bc + ad)
   		
	   
	   a[k] = a[k] + A[i] * c - B[i] * d ;
	   b[k] = b[k] + A[i] * d + B[i] * c ;

	}
 
  
	  a[k]=a[k]/N;
	  b[k]=b[k]/N;


    }

}