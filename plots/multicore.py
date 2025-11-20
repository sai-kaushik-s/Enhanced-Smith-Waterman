import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Global style
# -------------------------
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

# -------------------------
# Raw data (N = 16384)
# -------------------------
N = 16384
threads = np.array([1, 2, 4, 8, 16, 24, 48, 64, 128])

t_py = np.array([
    79.197979, 78.689244, 78.213922, 77.630084, 77.214162,
    76.307094, 73.507969, 68.777377, 77.472233
])

t_cpp = np.array([
    0.898253, 0.756152, 0.743886, 0.740282, 0.744330,
    0.896428, 0.763765, 0.745992, 0.875640
])

t_opt = np.array([
    0.353188, 0.180059, 0.0941034, 0.060815, 0.0386199,
    0.0379184, 0.737694, 1.00679, 0.890691
])

# 필요하면 1~24 threads까지만 사용
mask = threads <= 128
threads_plot = threads[mask]
t_py_plot  = t_py[mask]
t_cpp_plot = t_cpp[mask]
t_opt_plot = t_opt[mask]

def gcups(n, t):
    return (n * n) / (t * 1e9)

GCUPS_cpp = gcups(N, t_cpp_plot)
GCUPS_opt = gcups(N, t_opt_plot)

# Speedup vs Python baseline
speedup_cpp = t_py_plot / t_cpp_plot
speedup_opt = t_py_plot / t_opt_plot

# -------------------------
# Plot
# -------------------------
x = np.arange(len(threads_plot))
bar_width = 0.4

fig, ax1 = plt.subplots(figsize=(7, 3.5))

# GCUPS bars (left y-axis)
ax1.bar(x - bar_width/2, GCUPS_cpp, width=bar_width, label='C++ scalar')
ax1.bar(x + bar_width/2, GCUPS_opt, width=bar_width, label='C++ optimized')

ax1.set_xlabel('Thread count (#)')
ax1.set_ylabel('Throughput (GCUPS)')
ax1.set_xticks(x)
ax1.set_xticklabels(threads_plot)
ax1.grid(axis='y', linestyle='--', alpha=0.3)

# Speedup points (right y-axis, no lines)
ax2 = ax1.twinx()
ax2.scatter(x - bar_width/2, speedup_cpp,
            color='red', marker='x',
            label='Speedup: scalar vs Python')
ax2.scatter(x + bar_width/2, speedup_opt,
            color='black', marker='x',
            label='Speedup: optimized vs Python')
ax2.set_ylabel('Speedup')

# Merge legends, move to left
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
legend = ax1.legend(h1 + h2, l1 + l2,
                    loc='lower center',
                    bbox_to_anchor=(0.5, 1.05),
                    ncol=2,
                    frameon=False)

plt.tight_layout()
plt.savefig('sw_scaling_gcups_speedup.pdf', dpi=300)
plt.show()
