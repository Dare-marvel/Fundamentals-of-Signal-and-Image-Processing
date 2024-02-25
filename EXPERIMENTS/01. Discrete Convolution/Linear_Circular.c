#include <stdio.h>

int main()
{
    int i, j, r, m, n;

    // printf("LINEAR CONVOLUTION USING CIRCULAR CONVOLUTION\n\n\n");
    printf("Enter L, M: ");
    scanf("%d %d", &m, &n);

    int x1[10], x2[10], y[10];
    r = m + n - 1;

    for (i = 0; i < 10; i++)
    {
        x1[i] = 0;
        x2[i] = 0;
        y[i] = 0;
    }

    printf("Enter values of x[n]: ");
    for (i = 0; i < m; i++)
    {
        scanf("%d", &x1[i]);
    }

    printf("Enter values of y[n]: ");
    for (j = 0; j < n; j++)
    {
        scanf("%d", &x2[j]);
    }

    // PADDING LOGIC
    for (i = n; i < r; i++)
    {
        x2[i] = 0;
    }

    for (i = m; i < r; i++)
    {
        x1[i] = 0;
    }

    printf("x[n] = ");
    for (i = 0; i < r; i++)
    {
        printf("%d, ", x1[i]);
    }
    printf("\n");

    printf("h[n] = ");
    for (i = 0; i < r; i++)
    {
        printf("%d, ", x2[i]);
    }
    printf("\n");

    // LOGIC OF THE CODE USING ANALYTICAL FORMULA
    for (i = 0; i < r; i++)
    {
        for (j = 0; j < r; j++)
        {
            if (i - j < 0)
            {
                y[i] += x1[j] * x2[r + i - j];
            }
            else
            {
                y[i] += x1[j] * x2[i - j];
            }
        }
    }

    // DISPLAY OUTPUT MATRIX
    printf("y[n] = ");
    for (i = 0; i < r; i++)
    {
        printf("%d, ", y[i]);
    }

    return 0;
}
