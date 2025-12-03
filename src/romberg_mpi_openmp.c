/*
 * ============================================================================
 * Hybrid MPI + OpenMP Romberg Integration (Heavy Integrand)
 * HPC4DS Final Project – University of Trento
 * Authors: Lewis Ndambiri & Mehrab Fajar
 *
 * Description:
 *   Hybrid parallel implementation of Romberg integration using:
 *     - MPI for distributing trapezoidal subintervals
 *     - OpenMP for parallelizing the heavy integrand inside each process
 *
 *   The integrand is intentionally computationally expensive (HEAVY_ITERS)
 *   to test hybrid performance.
 *
 *   MPI_Reduce is used because:
 *     - Only rank 0 fills the Romberg table.
 *     - Other ranks only supply partial trapezoid sums.
 *
 * Compilation:
 *   mpicc -fopenmp -O3 -std=c99 romberg_mpi_openmp.c -lm -o romberg_mpi_openmp
 *
 * Execution example:
 *   export OMP_NUM_THREADS=2
 *   mpirun -np 4 ./romberg_mpi_openmp
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

// Portable definition of PI
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_LEVELS 20
#define HEAVY_ITERS 1000000



/* Heavy integrand function */
double f(double x) {
    double acc = 0.0;
    #pragma omp parallel for reduction(+ : acc)
    for (int i = 0; i < HEAVY_ITERS; i++) {
        acc += sin(x) * exp(-x * x);
    }
    return acc;
}
/* ----------------------------------------------------------------------------
 * Hybrid Trapezoidal Rule
 *
 * MPI:
 *   - Distributes subintervals using block partition + remainder.
 *
 * OpenMP:
 *   - Parallelizes sum of local function evaluations using reduction.
 *
 * MPI_Reduce:
 *   - Only rank 0 needs the global trapezoidal result.
 * -------------------------------------------------------------------------- */

double trapezoidal_hybrid(double a, double b, int n, int rank, int size) {
    double h = (b - a) / n;

    // Distribute n intervals evenly + 1 extra for first 'remainder' ranks
    int base = n / size;
    int remainder = n % size;
    int local_n = base + (rank < remainder ? 1 : 0);
    int start_idx = rank * base + (rank < remainder ? rank : remainder);
    double local_a = a + start_idx * h;

    double local_sum = 0.0;

    // Parallelize trapezoid summation inside each MPI process
    #pragma omp parallel for reduction(+ : local_sum) schedule(static)
    for (int i = 1; i < local_n; i++) {
        local_sum += f(local_a + i * h);
    }

    // Add endpoints for local interval
    double left  = f(local_a);
    double right = f(local_a + local_n * h);
    local_sum += 0.5 * (left + right);
    local_sum *= h;

    // Reduce to rank 0
    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return global_sum;
}
/* ----------------------------------------------------------------------------
 * MAIN: Hybrid MPI + OpenMP Romberg Integration
 * -------------------------------------------------------------------------- */
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int max_level = MAX_LEVELS;
    if (argc > 1) {
        max_level = atoi(argv[1]);
        if (max_level > MAX_LEVELS) max_level = MAX_LEVELS;
    }

    const double a = 0.0, b = M_PI;
    double R[MAX_LEVELS + 1][MAX_LEVELS + 1] = {{0.0}};

    if (rank == 0) {
        printf("=== Hybrid MPI + OpenMP Romberg ===\n");
        printf("MPI ranks          : %d\n", size);
        printf("OMP threads/rank   : %d\n", omp_get_max_threads());
        printf("Max Romberg level  : %d\n", max_level);
        printf("----------------------------------\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    /* Build the Romberg table (rank 0 only does extrapolation) */

    for (int lev = 0; lev <= max_level; lev++) {
        int n = 1 << lev;
        if (n < size) continue;

        double T = trapezoidal_hybrid(a, b, n, rank, size);

        if (rank == 0) {
            R[lev][0] = T;
            for (int j = 1; j <= lev; j++) {
                double factor = pow(4.0, j);
                R[lev][j] = R[lev][j-1] + (R[lev][j-1] - R[lev-1][j-1]) / (factor - 1.0);
            }
            printf("Completed level %2d (n=%d)\n", lev, n);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    
    if (rank == 0) {

        double estimate = R[max_level][max_level];
        printf("\n");
        printf("Romberg Estimate   : %.12e\n", estimate);
        printf("Total Time         : %.6f seconds\n", end - start); 
    }

/*   For fair comparison, we do not check for below
 if (rank == 0) {
        double estimate = R[max_level][max_level];

        // Reference value (same as serial)
        double ref = 0.0;
        int ref_n = 1000000;
        double h = (b - a) / ref_n;
        for (int i = 0; i <= ref_n; i++) {
            double x = a + i * h;
            double w = (i == 0 || i == ref_n) ? 0.5 : 1.0;
            ref += w * sin(x) * exp(-x * x);
        }
        ref *= h * HEAVY_ITERS;

        printf("\nRomberg Estimate   : %.12e\n", estimate);
        printf("Reference Integral : %.12e\n", ref);
        printf("Absolute Error     : %.6e\n", fabs(estimate - ref));
        printf("Total Time         : %.6f seconds\n", end - start);
    }
*/
    MPI_Finalize();
    return 0;
}
