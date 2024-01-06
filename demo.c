// Simple program that keeps multiple CPUs busy
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sched.h>
#include <time.h>

int main(int argc, char* argv[]) {
    assert(argc == 3);
    int n_threads = atoi(argv[1]);
    omp_set_num_threads(n_threads);
    long n_per_thread = atol(argv[2]);
    printf("Running with %d threads, %ld iterations on each\n", n_threads, n_per_thread);
    clock_t clock_time = clock();
    double wtime = omp_get_wtime();
    #pragma omp parallel for
    for (int t = 0; t < n_threads; t++) {
        long s = 0, x = 0, a = 0, o = 0;
        for (long ii = 0; ii < n_per_thread; ii++) {
            long i = n_per_thread * (long)t + ii;
            s += i;
            x ^= i;
            a &= i;
            o |= i;
        }
    }
    printf("Elapsed clock time: %lf s\n", ((double)(clock() - clock_time)) / CLOCKS_PER_SEC);
    printf("Elapsed wtime: %lf s\n", omp_get_wtime() - wtime);
    return 0;
}