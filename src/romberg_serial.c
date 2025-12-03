/*
 * ============================================================================
 * Serial Romberg Integration
 * HPC4DS Final Project – University of Trento
 * Authors: Lewis Ndambiri & Mehrab Fajar
 *
 * PURPOSE:
 *   - This is the *true* serial baseline for benchmarking MPI and hybrid versions.
 *   - NO MPI calls.
 *   - Computation is intentionally heavy to simulate HPC workloads.
 *
 * RUN:
 *     ./romberg_serial 
 *
 * NOTES:
 *   - You may enable/disable the expensive reference computation at the bottom.
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>   // clock_gettime for accurate serial timing

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_LEVELS 20
#define HEAVY_ITERS 1000000        // artificial computational load

/* ----------------------------------------------------------------------------
 * Heavy integrand
 *   f(x) = sin(x)*exp(-x^2), repeated HEAVY_ITERS times
 *
 *   This simulates a computationally expensive integrand typically found in
 *   HPC applications (e.g., PDE solvers, physics simulations, etc.)
 * ----------------------------------------------------------------------------
 */
double f(double x) {
    double base = sin(x) * exp(-x * x);
    double s = 0.0;

    for (int i = 0; i < HEAVY_ITERS; i++)
        s += base;

    return s;
}

/* ----------------------------------------------------------------------------
 * Serial trapezoidal rule
 *   Computes integral of f(x) from a to b using n subintervals.
 * ----------------------------------------------------------------------------
 */
double trapezoidal(double a, double b, int n) {
    double h = (b - a) / n;             // step size
    double sum = (f(a) + f(b)) / 2.0;   // boundary terms

    for (int i = 1; i < n; i++) {
        sum += f(a + i * h);
    }

    return sum * h;
}

/* ----------------------------------------------------------------------------
 * High-resolution wall clock timer
 * ----------------------------------------------------------------------------
 */
double walltime() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * t.tv_nsec;
}

/* ----------------------------------------------------------------------------
 * MAIN PROGRAM
 * ----------------------------------------------------------------------------
 */
int main(int argc, char** argv) {

    /* Determine number of Romberg refinement levels */
    int max_level = MAX_LEVELS;
    if (argc > 1) {
        max_level = atoi(argv[1]);
        if (max_level > MAX_LEVELS) max_level = MAX_LEVELS;
    }

    const double a = 0.0;
    const double b = M_PI;

    /* Romberg table R[i][j]
     *  i = trapezoid refinement level
     *  j = Richardson extrapolation level
     */
    double R[MAX_LEVELS + 1][MAX_LEVELS + 1] = {{0.0}};

    printf("\n=== Serial Romberg Integration (Heavy Integrand) ===\n");
    printf("Max Levels: %d\n", max_level);
    printf("HEAVY_ITERS: %d\n", HEAVY_ITERS);

    double start = walltime();

    /* Construct Romberg table */
    for (int i = 0; i <= max_level; i++) {
        int n = 1 << i;        // number of intervals = 2^i

        /* Compute trapezoidal rule for this refinement level */
        R[i][0] = trapezoidal(a, b, n);

        /* Richardson extrapolation */
        for (int j = 1; j <= i; j++) {
            double factor = pow(4.0, j);
            R[i][j] = R[i][j - 1] +
                      (R[i][j - 1] - R[i - 1][j - 1]) / (factor - 1.0);
        }

        printf("Level %2d completed (n = %d)\n", i, n);
    }

    double end = walltime();
    double estimate = R[max_level][max_level];

    printf("\nFinal Romberg Estimate : %.12e\n", estimate);
    printf("Total Time (serial)    : %.6f seconds\n", end - start);

    /* ------------------------------------------------------------------------
     * OPTIONAL: Compute HIGH-PRECISION REFERENCE VALUE
     *
     * WARNING:
     *   - This uses 1,000,000 trapezoid steps
     *   - Each step calls f(x) which repeats HEAVY_ITERS times
     *   - Very slow! Only enable for testing correctness.
     * ------------------------------------------------------------------------
     */

#if 0   // Change to 0/1 to disable/enable reference computation

    printf("\nComputing reference value... (may take a long time!)\n");

    int ref_n = 1000000;
    double h = (b - a) / ref_n;
    double ref = 0.0;

    for (int i = 0; i <= ref_n; i++) {
        double x = a + i * h;
        double w = (i == 0 || i == ref_n) ? 0.5 : 1.0;
        ref += w * sin(x) * exp(-x * x);
    }

    ref *= h * HEAVY_ITERS;

    printf("Reference Value        : %.12e\n", ref);
    printf("Absolute Error         : %.6e\n", fabs(estimate - ref));

#endif

    return 0;
}
