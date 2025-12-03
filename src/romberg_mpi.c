/*
 * ============================================================================
 * Parallel Romberg Integration (MPI Version)
 * HPC4DS Final Project – University of Trento
 * Authors: Lewis Ndambiri & Mehrab Fajar
 *
 * PURPOSE:
 *   - Parallelize the trapezoidal rule using MPI.
 *   - Perform Romberg extrapolation on rank 0.
 *   - Heavy integrand simulates expensive HPC calculations.
 *
 * MATCHES SERIAL VERSION:
 *   - Same integrand
 *   - Same Romberg structure
 *   - Same trapezoidal logic
 *   - Same heavy workload per f(x)
 *
 * RUN:
 *     mpiexec -n 4 ./romberg_mpi 
 *
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define HEAVY_ITERS 1000000
#define MAX_LEVELS 20

/* ---------------------------------------------------------------------------
 * Heavy integrand f(x)
 * --------------------------------------------------------------------------- */
double f(double x) {
    double base = sin(x) * exp(-x * x);
    double s = 0.0;

    for (int i = 0; i < HEAVY_ITERS; i++)
        s += base;

    return s;
}

/* ---------------------------------------------------------------------------
 * Parallel trapezoidal rule using MPI_Reduce
 *Parameters:
 *   a, b     - Integration bounds
 *   n        - Number of trapezoidal subintervals (n = 2^level)
 *   my_rank  - Rank of this process
 *   comm_sz  - Total number of MPI processes
 *
 * Each process computes a subset of the f(x) evaluations.
 * Rank 0 combines them into the global result.
 * --------------------------------------------------------------------------- */
double parallel_trapezoid(double a, double b, int n, int my_rank, int comm_sz) {

    double h = (b - a) / n;

    // Split the n subintervals across processes as evenly as possible:
    int base = n / comm_sz;
    int remainder = n % comm_sz;

    int local_n = base + (my_rank < remainder);
    int local_start = my_rank * base + (my_rank < remainder ? my_rank : remainder);

    double local_a = a + local_start * h;
    double local_b = local_a + local_n * h;

    // Local trapezoid sum
    double local_sum = (f(local_a) + f(local_b)) / 2.0;

    for (int i = 1; i < local_n; i++)
        local_sum += f(local_a + i * h);

    local_sum *= h;

    // Combine partial sums on rank 0
    double global_sum = 0.0;

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    return global_sum;
}

/* ---------------------------------------------------------------------------
 * MAIN MPI PROGRAM
 * --------------------------------------------------------------------------- */
int main(int argc, char* argv[]) {
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    int max_levels = MAX_LEVELS;
    if (argc > 1) {
        max_levels = atoi(argv[1]);
        if (max_levels > MAX_LEVELS) max_levels = MAX_LEVELS;
    }
    // Integration bounds
    const double a = 0.0, b = M_PI;

    // Romberg table allocated only by rank 0
    double **R = NULL;
    if (my_rank == 0) {
        R = malloc((max_levels + 1) * sizeof(double*));
        for (int i = 0; i <= max_levels; i++)
            R[i] = calloc(max_levels + 1, sizeof(double));
    }
    // Start timing
    double start_time = 0.0;
    if (my_rank == 0)
        start_time = MPI_Wtime();

    // Build the Romberg table (parallel trapezoid + serial extrapolation)
    for (int i = 0; i <= max_levels; i++) {
        int n = 1 << i;

        double T = parallel_trapezoid(a, b, n, my_rank, comm_sz);
        
        // Romberg extrapolation (rank 0 only)
        if (my_rank == 0) {
            R[i][0] = T;

            // Richardson extrapolation formula:
            // R[i][j] = R[i][j-1] + (R[i][j-1] - R[i-1][j-1]) / (4^j - 1)
            
            for (int j = 1; j <= i; j++) {
                double factor = pow(4.0, j);
                R[i][j] = R[i][j - 1] +
                         (R[i][j - 1] - R[i - 1][j - 1]) / (factor - 1.0);
            }

            printf("Level %2d complete (n = %d)\n", i, n);
        }
    }
    // Stop timing
    if (my_rank == 0) {
        double end_time = MPI_Wtime();

        printf("\n=== MPI Romberg Integration Summary ===\n");
        printf("Integrand        : sin(x) * exp(-x^2)\n");
        printf("Integration Range: [%.2f, %.2f]\n", a, b);
        printf("Processes       : %d\n", comm_sz);
        printf("Max Levels      : %d\n", max_levels);
        printf("Final Estimate  : %.12e\n", R[max_levels][max_levels]);
        printf("Total Time      : %.6f seconds\n", end_time - start_time);
        printf("========================================\n");
        
        // Cleanup
        for (int i = 0; i <= max_levels; i++)
            free(R[i]);
        free(R);
    }

    MPI_Finalize();
    return 0;
}
