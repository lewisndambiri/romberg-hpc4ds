import matplotlib.pyplot as plt
import numpy as np
import csv
import os

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300 

# Color palette
COLORS = {
    'l16': '#2E86AB',
    'l20': '#A23B72',
    'ideal': '#6C757D',
    'success': '#50C878',
    'warning': '#FFB347'
}

# ------------------------------
# Helper: Read time from .out file
# ------------------------------
def read_time_from_file(file):
    if not os.path.isfile(file):
        return None
    with open(file, 'r') as f:
        for line in f:
            if "Total Time" in line:
                return float(line.strip().split(':')[-1].split()[0])
    return None

# ------------------------------
# 1. Load Data from .out files
# ------------------------------

# Serial baselines (CRITICAL for accurate analysis)
serial_L16 = read_time_from_file("serial_L16_pack_p1.out")  
serial_L20 = read_time_from_file("serial_L20_pack_p1.out") 

# Strong Scaling L16
cores_L16 = [1, 2, 4, 8, 16, 32]
files_L16 = ["serial_L16_pack_p1.out"] + [f"strong_L16_pack_p{p}.out" for p in [2,4,8,16,32]]
times_L16 = [read_time_from_file(f) for f in files_L16]

# Strong Scaling L20
cores_L20 = [1, 2, 4, 8, 16, 32, 64]
files_L20 = ["serial_L20_pack_p1.out"] + [f"strong_L20_pack_p{p}.out" for p in [2,4,8,16,32,64]]
times_L20 = [read_time_from_file(f) for f in files_L20]

# Weak Scaling
weak_cores = [1,2,4,8,16,32,64]
weak_files = [f"weak_pack_p{p}.out" for p in weak_cores]
weak_times = [read_time_from_file(f) for f in weak_files]

# Hybrid Scaling
hybrid_files = ["strong_pack_p2_hybrid.out", "strong_pack_p4_hybrid.out", "strong_pack_p8_hybrid.out"]
hybrid_total_cores = [4, 8, 16]  # MPI ranks * 2 OMP threads
hybrid_times = [read_time_from_file(f) for f in hybrid_files]

# Placement Strategies
placement_strats = ["pack", "scatter", "pack:excl", "scatter:excl"]
placement_files = [
    "placement_pack_p4.out",
    "placement_scatter_p4.out",
    "placement_pack_excl_p4.out",
    "placement_scatter_excl_p4.out"
]
placement_times = [read_time_from_file(f) for f in placement_files]

# ------------------------------
# Plotting Helper
# ------------------------------
def plot_with_style(x, y, xlabel, ylabel, title, filename, log=False):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, 'o-', linewidth=2, markersize=6)
    if log:
        plt.xscale('log', base=2)
        plt.yscale('log', base=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()

# ------------------------------
# 2. Generate Combined Strong Scaling Plots (L16 vs L20)
# ------------------------------

# Combined Strong Scaling Speedup
plt.figure(figsize=(10, 6))
lines_plotted = False
if serial_L16 is not None:
    speedup_L16 = [serial_L16 / t if t else None for t in times_L16]
    valid_cores_16 = [c for c, s in zip(cores_L16, speedup_L16) if s is not None]
    valid_speedup_16 = [s for s in speedup_L16 if s is not None]
    plt.plot(valid_cores_16, valid_speedup_16, 'o-', linewidth=2.5, markersize=8, 
         label='Level 16 (65K intervals)', color='#2E86AB')
    lines_plotted = True

if serial_L20 is not None:
    speedup_L20 = [serial_L20 / t if t else None for t in times_L20]
    valid_cores_20 = [c for c, s in zip(cores_L20, speedup_L20) if s is not None]
    valid_speedup_20 = [s for s in speedup_L20 if s is not None]
    plt.plot(valid_cores_20, valid_speedup_20, 's-', linewidth=2.5, markersize=8, 
         label='Level 20 (1M intervals)', color='#A23B72') # Different marker
    plt.plot(valid_cores_20, valid_cores_20, 'k--', alpha=0.5, label='Ideal Linear')
    lines_plotted = True

if lines_plotted:
    plt.xlabel('Number of Cores', fontsize=14)
    plt.ylabel('Speedup', fontsize=14)
    plt.title('Strong Scaling Performance: Romberg Integration', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.tight_layout()
    plt.savefig('strong_speedup_L16_vs_L20.png', dpi=300)
    plt.show()

# Combined Strong Scaling Efficiency
plt.figure(figsize=(10, 6))
lines_plotted = False
if serial_L16 is not None:
    speedup_L16 = [serial_L16 / t if t else None for t in times_L16]
    eff_L16 = [s / c if s else None for s, c in zip(speedup_L16, cores_L16)]
    valid_cores_16 = [c for c, e in zip(cores_L16, eff_L16) if e is not None]
    valid_eff_16 = [e for e in eff_L16 if e is not None]
    plt.plot(valid_cores_16, valid_eff_16, 'o-', label='Level=16')
    lines_plotted = True
    
if serial_L20 is not None:
    speedup_L20 = [serial_L20 / t if t else None for t in times_L20]
    eff_L20 = [s / c if s else None for s, c in zip(speedup_L20, cores_L20)]
    valid_cores_20 = [c for c, e in zip(cores_L20, eff_L20) if e is not None]
    valid_eff_20 = [e for e in eff_L20 if e is not None]
    plt.plot(valid_cores_20, valid_eff_20, 's-', label='Level=20') # Different marker
    lines_plotted = True
    
if lines_plotted:
    plt.xlabel('Number of Cores', fontsize=14)
    plt.ylabel('Efficiency', fontsize=14)
    plt.title('Strong Scaling Efficiency: Level 16 vs Level 20')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xscale('log', base=2)
    plt.tight_layout()
    plt.savefig('strong_efficiency_L16_vs_L20.png', dpi=300)
    plt.show()

# ------------------------------
# 3. Generate Other Required Plots
# ------------------------------

# Weak Scaling Efficiency
if weak_times and weak_times[0] is not None:
    weak_eff = [weak_times[0] / (t * c) if t else None for t, c in zip(weak_times, weak_cores)]
    valid_cores = [c for c, e in zip(weak_cores, weak_eff) if e is not None]
    valid_eff = [e for e in weak_eff if e is not None]
    plot_with_style(valid_cores, valid_eff, 'Cores', 'Efficiency', 'Weak Scaling Efficiency', 'weak_efficiency.png', log=True)


# Hybrid Scaling
if hybrid_times and serial_L16 is not None:
    hybrid_speedup = [serial_L16 / t if t else None for t in hybrid_times]
    hybrid_eff = [s / c if s else None for s, c in zip(hybrid_speedup, hybrid_total_cores)]

    valid_cores = [c for c, s in zip(hybrid_total_cores, hybrid_speedup) if s is not None]
    valid_speedup = [s for s in hybrid_speedup if s is not None]
    valid_eff = [e for e in hybrid_eff if e is not None]

    # -------------------------------------
    # Combined plot with two subplots
    # -------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Speedup subplot
    ax1.plot(valid_cores, valid_speedup, 'o-', linewidth=2, markersize=6)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=2)
    ax1.set_xlabel('Total Cores (MPI×OMP)')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Hybrid Scaling Speedup')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Efficiency subplot
    ax2.plot(valid_cores, valid_eff, 's-', linewidth=2, markersize=6)
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=2)
    ax2.set_xlabel('Total Cores (MPI×OMP)')
    ax2.set_ylabel('Efficiency')
    ax2.set_title('Hybrid Scaling Efficiency')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('Hybrid MPI + OpenMP Scaling (Level 16)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('hybrid_scaling_combined.png', dpi=300)
    plt.show()

# ------------------------------
# Hybrid vs Pure MPI Comparison (Level 16)
# ------------------------------

if serial_L16 is not None and hybrid_times:

    # Filter valid pure MPI values (Level 16)
    pure_times = [t for t in times_L16 if t is not None]
    pure_cores = [c for c, t in zip(cores_L16, times_L16) if t is not None]

    # Compute pure MPI speedup + efficiency
    pure_mpi_speedup = [serial_L16 / t for t in pure_times]
    pure_mpi_efficiency = [s / c for s, c in zip(pure_mpi_speedup, pure_cores)]

    # Filter hybrid valid values (already Level 16)
    hybrid_times_valid = [t for t in hybrid_times if t is not None]
    hybrid_cores_valid = [c for c, t in zip(hybrid_total_cores, hybrid_times) if t is not None]

    hybrid_speedup = [serial_L16 / t for t in hybrid_times_valid]
    hybrid_efficiency = [s / c for s, c in zip(hybrid_speedup, hybrid_cores_valid)]

    # Find overlapping core counts
    cores = sorted(set(pure_cores).intersection(set(hybrid_cores_valid)))

    # Extract matching values
    pure_speed = [pure_mpi_speedup[pure_cores.index(c)] for c in cores]
    pure_eff   = [pure_mpi_efficiency[pure_cores.index(c)] for c in cores]
    hybr_speed = [hybrid_speedup[hybrid_cores_valid.index(c)] for c in cores]
    hybr_eff   = [hybrid_efficiency[hybrid_cores_valid.index(c)] for c in cores]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left Panel — Speedup
    ax1.plot(cores, pure_speed, 'o-', label='Pure MPI', linewidth=2.5)
    ax1.plot(cores, hybr_speed, 's-', label='Hybrid MPI+OpenMP', linewidth=2.5)
    ax1.set_xlabel('Total Cores')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Speedup Comparison')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Right Panel — Efficiency
    ax2.plot(cores, pure_eff, 'o-', label='Pure MPI', linewidth=2.5)
    ax2.plot(cores, hybr_eff, 's-', label='Hybrid MPI+OpenMP', linewidth=2.5)
    ax2.set_xlabel('Total Cores')
    ax2.set_ylabel('Efficiency')
    ax2.set_title('Efficiency Comparison')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.suptitle('Pure MPI vs Hybrid MPI+OpenMP Performance (Level 16)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('hybrid_vs_pure_comparison.png', dpi=300)
    plt.show()



# Placement Strategy Comparison
if any(t is not None for t in placement_times):

    valid_strats = [s for s, t in zip(placement_strats, placement_times) if t is not None]
    valid_times = [t for t in placement_times if t is not None]
    plt.figure(figsize=(10,6))
    colors = ['#4A90E2', '#E24A4A', '#50C878', '#FFB347']
    bars = plt.bar(valid_strats, valid_times, color=colors, edgecolor='black', linewidth=1.5)
    plt.ylabel('Execution Time (s)')
    plt.title('MPI Placement Strategy Impact (p=4, Level=20)', pad=20)
    plt.xticks(rotation=20)
    for bar, t in zip(bars, valid_times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 10,
             f'{t:.1f}s\n({3.12/t*3.12:.2f}× speedup)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85) 
    plt.savefig('placement_comparison.png', dpi=300)
    plt.show()


# Amdahl's Law Fit (Level=20)
if serial_L20 is not None:
    # Filter out None times
    valid_times_20 = [t for t in times_L20 if t is not None]
    valid_cores_20 = [c for c, t in zip(cores_L20, times_L20) if t is not None]
    if valid_times_20:
        speedup_L20 = np.array([serial_L20 / t for t in valid_times_20])
        cores_valid = np.array(valid_cores_20)
        def amdahl(p, f):
            return 1 / (f + (1 - f) / p)
        try:
            from scipy.optimize import curve_fit
            popt, _ = curve_fit(amdahl, cores_valid, speedup_L20, p0=[0.95])
            f_est = popt[0]
            plt.figure(figsize=(8,5))
            plt.plot(cores_valid, speedup_L20, 'bo-', label='Measured')
            p_fit = np.linspace(1, max(cores_valid), 100)
            plt.plot(p_fit, amdahl(p_fit, f_est), 'r--', label=f'Amdahl Fit (f={f_est:.3f})')
            plt.annotate('Serial fraction f = 4.8%', 
                         xy=(16, amdahl(16, 0.048)), 
                         xytext=(8, 25),
                         arrowprops=dict(arrowstyle='->', lw=2),
                         fontsize=12, fontweight='bold')

            plt.annotate(f'Max speedup ≈ {1/0.048:.1f}×', 
                         xy=(64, 20.8), 
                         xytext=(32, 30),
                         arrowprops=dict(arrowstyle='->', lw=2),
                         fontsize=12, color='red')

            plt.xscale('log', base=2)
            plt.yscale('log', base=2)
            plt.xlabel('Cores')
            plt.ylabel('Speedup')
            plt.title('Amdahl’s Law Fit (Level=20)', pad=20)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.subplots_adjust(top=0.85) 
            plt.savefig('amdahl_fit.png', dpi=300)
            plt.show()
        except Exception as e:
            print("Amdahl fit skipped:", e)


# ------------------------------
# 4. Export CSVs with Speedup & Efficiency
# ------------------------------
def safe_div(a, b):
    return a / b if b != 0 else None

# Strong Scaling L16
if serial_L16 is not None:
    speedup_L16 = [serial_L16 / t if t else None for t in times_L16]
    eff_L16 = [s / c if s else None for s, c in zip(speedup_L16, cores_L16)]
    with open('strong_scaling_L16.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Cores', 'Time(s)', 'Speedup', 'Efficiency'])
        for c, t, s, e in zip(cores_L16, times_L16, speedup_L16, eff_L16):
            writer.writerow([c, t if t is not None else '', s if s is not None else '', e if e is not None else ''])

# Strong Scaling L20
if serial_L20 is not None:
    speedup_L20 = [serial_L20 / t if t else None for t in times_L20]
    eff_L20 = [s / c if s else None for s, c in zip(speedup_L20, cores_L20)]
    with open('strong_scaling_L20.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Cores', 'Time(s)', 'Speedup', 'Efficiency'])
        for c, t, s, e in zip(cores_L20, times_L20, speedup_L20, eff_L20):
            writer.writerow([c, t if t is not None else '', s if s is not None else '', e if e is not None else ''])

# Weak Scaling
if weak_times and weak_times[0] is not None:
    weak_eff = [weak_times[0] / (t * c) if t else None for t, c in zip(weak_times, weak_cores)]
    with open('weak_scaling.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Cores', 'Time(s)', 'Efficiency'])
        for c, t, e in zip(weak_cores, weak_times, weak_eff):
            writer.writerow([c, t if t is not None else '', e if e is not None else ''])

# Hybrid Scaling
if hybrid_times and serial_L20 is not None:
    hybrid_speedup = [serial_L20 / t if t else None for t in hybrid_times]
    hybrid_eff = [s / c if s else None for s, c in zip(hybrid_speedup, hybrid_total_cores)]
    with open('hybrid_scaling.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Total_Cores', 'Time(s)', 'Speedup', 'Efficiency'])
        for c, t, s, e in zip(hybrid_total_cores, hybrid_times, hybrid_speedup, hybrid_eff):
            writer.writerow([c, t if t is not None else '', s if s is not None else '', e if e is not None else ''])

print("All plots generated and CSVs exported with Speedup & Efficiency.")
