#include <unistd.h>
#include <stdlib.h>
#include <stdlib>
#include <omp.h>
#include <stdio.h>

#define THREADS 4
#define N 16

int main ( ) {
  #pragma omp parallel for schedule(dynamic) collapse(2)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      sleep(j);
      printf("Thread %d has completed iteration = %d, %d.\n",
      omp_get_thread_num( ), i, j);
    }
  }

  printf("All done!\n");
  return 0;
}