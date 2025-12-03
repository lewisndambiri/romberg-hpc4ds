# Makefile for Romberg HPC4DS Project
# Authors: Lewis Ndambiri & Mehrab Fajar
# Builds serial, MPI, and hybrid versions

CC = mpicc
# Add _POSIX_C_SOURCE=199309L to enable clock_gettime()
CFLAGS = -std=c99 -O3 -Wall -lm -D_POSIX_C_SOURCE=199309L
OMPFLAGS = -fopenmp

# Targets
SERIAL = romberg_serial
MPI = romberg_mpi
HYBRID = romberg_mpi_openmp

.PHONY: all serial mpi hybrid clean

all: serial mpi hybrid

serial: src/$(SERIAL).c
	$(CC) $(CFLAGS) -o $(SERIAL) $<

mpi: src/$(MPI).c
	$(CC) $(CFLAGS) -o $(MPI) $<

hybrid: src/$(HYBRID).c
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $(HYBRID) $<

clean:
	rm -f $(SERIAL) $(MPI) $(HYBRID) *.out *.err
