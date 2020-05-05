#include <stdio.h>
#include <omp.h>

int main()
{
    int count = omp_get_max_threads();
    printf("Count (%d)", count);
}
