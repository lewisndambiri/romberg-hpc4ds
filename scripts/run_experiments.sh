#!/bin/bash
# run_experiments_all.sh — Fully automated submission for all PBS experiments
set -e

echo "Starting Romberg HPC4DS experiments..."
module load mpich-3.2

echo "Building..."
make clean && make

#-------------------------------------------------------------------
# Cluster-specific parameters
#-------------------------------------------------------------------
CORES_PER_NODE=48
MEM_PER_PROC=2

#-------------------------------------------------------------------
# Helper function
#-------------------------------------------------------------------
submit_job(){
    local script=$1      # e.g., romberg_mpi.pbs
    local p=$2
    local level=$3
    local placement=$4   # e.g., "pack:excl"
    local scaling=$5     # e.g., "strong", "placement", "strong_L20"
    local hybrid=$6      # "yes"/"no"

    # Sanitize placement for filename (replace : with _)
    local placement_safe="${placement//:/_}"
    local output_name
    if [[ "$hybrid" == "yes" ]]; then
        output_name="${scaling}_${placement_safe}_p${p}_hybrid.out"
    else
        output_name="${scaling}_${placement_safe}_p${p}.out"
    fi

    # Node allocation logic
    local nodes_needed=1
    local ncpus_per_node=$p
    if [[ "$placement" == scatter* ]]; then
        nodes_needed=$p
        ncpus_per_node=1
    elif [[ $p -gt $CORES_PER_NODE ]]; then
        nodes_needed=$(( (p + CORES_PER_NODE - 1) / CORES_PER_NODE ))
        ncpus_per_node=$(( (p + nodes_needed - 1) / nodes_needed ))
    fi
    local mem_per_node=$(( ncpus_per_node * MEM_PER_PROC ))
    local select_line="select=${nodes_needed}:ncpus=${ncpus_per_node}:mem=${mem_per_node}gb"

    # Command line
    local exec_line
    if [[ "$hybrid" == "yes" ]]; then
        local omp_threads=2
        exec_line="module load mpich-3.2; export OMP_NUM_THREADS=${omp_threads}; mpiexec -n $p ./romberg_mpi_openmp $level"
    elif [[ "$script" == *"serial"* ]]; then
        exec_line="./romberg_serial $level"
    else
        exec_line="module load mpich-3.2; mpiexec -n $p ./romberg_mpi $level"
    fi

    # Generate PBS script from template
    sed "s|DUMMY\.out|${output_name}|g" "scripts/${script}" | \
    sed "s|DUMMY_EXEC_LINE|${exec_line}|g" | \
    sed "s|select=[^ ]*|${select_line}|g" | \
    sed "s|place=[^ ]*|place=${placement}|g" > ~/job.pbs

    job_id=$(qsub ~/job.pbs)
    echo " Job $job_id → $output_name"
}

#-------------------------------------------------------------------
# Serial baselines (L16 and L20)
#-------------------------------------------------------------------
for lvl in 16 20; do
    echo "Submitting serial (Level=$lvl)"
    submit_job "romberg_serial.pbs" 1 $lvl "pack" "serial_L${lvl}" "no"
done

#-------------------------------------------------------------------
# Strong scaling on multiple problem sizes: Level=16 (up to p=16) and Level=20 (up to p=64)
#-------------------------------------------------------------------
# Level 16
for p in 2 4 8 16 32; do
    echo "Submitting strong scaling (Level=16, p=$p)"
    submit_job "romberg_mpi.pbs" $p 16 "pack" "strong_L16" "no"
done
# Level 20
for p in 2 4 8 16 32 64; do
    echo "Submitting strong scaling (Level=20, p=$p)"
    submit_job "romberg_mpi.pbs" $p 20 "pack" "strong_L20" "no"
done

#-------------------------------------------------------------------
# Weak scaling 
#-------------------------------------------------------------------
declare -A LEVELS=([1]=12 [2]=13 [4]=14 [8]=15 [16]=16 [32]=17 [64]=18)
for p in 1 2 4 8 16 32 64; do
    level=${LEVELS[$p]}
    echo "Submitting weak scaling: p=$p, level=$level"
    submit_job "romberg_mpi.pbs" $p $level "pack" "weak" "no"
done

#-------------------------------------------------------------------
# Placement strategies
#-------------------------------------------------------------------
PLACEMENTS=("pack" "scatter" "pack:excl" "scatter:excl")
for place in "${PLACEMENTS[@]}"; do
    echo "Submitting placement: $place"
    submit_job "romberg_mpi.pbs" 4 20 "$place" "placement" "no"
done

#-------------------------------------------------------------------
# Hybrid MPI+OpenMP scaling
#-------------------------------------------------------------------
for p in 2 4 8; do
    echo "Submitting hybrid MPI+OpenMP: p=$p"
    submit_job "romberg_hybrid.pbs" $p 16 "pack" "strong" "yes"
done

echo ""
echo "All jobs submitted."
