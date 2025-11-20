import numpy as np
import matplotlib.pyplot as plt

# Make fonts bigger globally
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

# -------------------------
# Raw data
# -------------------------
N = np.array([64, 256, 1024, 4096, 16384])

t_python = np.array([0.001250, 0.014517, 0.236184, 3.931092, 65.869466])
t_cpp    = np.array([0.000061, 0.000572, 0.006348, 0.048682, 0.737224])
t_opt    = np.array([3.5712e-05, 0.000308463, 0.00396892, 0.0220596, 0.293656])

def gcups(n, t):
    return (n * n) / (t * 1e9)

GCUPS_python = gcups(N, t_python)
GCUPS_cpp    = gcups(N, t_cpp)
GCUPS_opt    = gcups(N, t_opt)

speedup_cpp = t_python / t_cpp
speedup_opt = t_python / t_opt

# -------------------------
# Plot
# -------------------------
x = np.arange(len(N))
bar_width = 0.3

fig, ax1 = plt.subplots(figsize=(8, 3.5))

# GCUPS bars (left y-axis)
ax1.bar(x - bar_width, GCUPS_python, width=bar_width, label='Python')
ax1.bar(x,             GCUPS_cpp,    width=bar_width, label='C++ scalar')
ax1.bar(x + bar_width, GCUPS_opt,    width=bar_width, label='C++ optimized')

ax1.set_xlabel('Sequence length $N$')
ax1.set_ylabel('Throughput (GCUPS)')
ax1.set_xticks(x)
ax1.set_xticklabels(N)
ax1.grid(axis='y', linestyle='--', alpha=0.3)

# Make axis ticks thicker/bigger if you like
ax1.tick_params(axis='both', which='major', length=6, width=1.2)
ax2 = ax1.twinx()
ax2.tick_params(axis='both', which='major', length=6, width=1.2)

# Speedup points (right y-axis, no lines)
ax2.scatter(x,             speedup_cpp, color='red', marker='x',
            label='Speedup: C++ scalar vs Python')
ax2.scatter(x + bar_width, speedup_opt, color='black', marker='x',
            label='Speedup: C++ opt vs Python')
ax2.set_ylabel('Speedup')

# Merge legends
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='upper left')

plt.tight_layout()
plt.savefig('sw_gcups_speedup_vs_N.pdf', dpi=300)
plt.show()
