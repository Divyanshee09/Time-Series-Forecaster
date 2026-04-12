"""Generate comparison bar charts for the DGSAR paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
})

# ── Data from Table 3 (MASE) ──
series = ['M4 Hourly', 'Daily (D1)', 'Bitcoin', 'COVID-19', 'Airline']
fs_light = [1.165, 0.455, 0.765, 0.546, 0.672]
prophet  = [3.298, 0.759, 3.150, 4.833, 5.201]
ets      = [1.709, 0.430, 0.725, 0.678, 0.739]
theta    = [1.154, 0.483, 0.667, 1.250, 0.732]
snav     = [0.980, 0.665, 1.039, 1.470, 1.134]

# ── Chart 1: MASE comparison bar chart ──
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(series))
w = 0.15

bars = [
    (fs_light, 'DGSAR', '#2563eb'),
    (prophet, 'Prophet', '#dc2626'),
    (ets, 'ETS', '#16a34a'),
    (theta, 'Theta', '#ca8a04'),
    (snav, 'Seasonal Naive', '#7c3aed'),
]
for i, (vals, label, color) in enumerate(bars):
    # Cap display at 2.5 for readability, mark truncated bars
    display_vals = [min(v, 2.5) for v in vals]
    rects = ax.bar(x + (i - 2) * w, display_vals, w, label=label, color=color, alpha=0.85)
    for j, (dv, rv) in enumerate(zip(display_vals, vals)):
        if rv > 2.5:
            ax.text(x[j] + (i - 2) * w, dv + 0.03, f'{rv:.1f}',
                    ha='center', va='bottom', fontsize=7, rotation=90)

ax.set_ylabel('MASE (lower is better)')
ax.set_title('MASE Comparison Across All Series')
ax.set_xticks(x)
ax.set_xticklabels(series)
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, label='MASE = 1 (Seasonal Naive baseline)')
ax.set_ylim(0, 2.8)
ax.legend(loc='upper right', ncol=2)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig('c:/Users/Swastik/Desktop/FutureScope/mase_comparison.png', bbox_inches='tight')
fig.savefig('c:/Users/Swastik/Desktop/FutureScope/mase_comparison.pdf', bbox_inches='tight')
print('Saved mase_comparison.png/pdf')
plt.close()

# ── Data from Table 2 (RMSE) ──
rmse_fs   = [102.5, 108.1, 16793, 83790, 38.3]
rmse_prop = [57.6, 100.2, 19325, 91808, 45.7]
rmse_ets  = [161.2, 589.1, 29059, 40713, 213.3]
rmse_theta= [46.6, 171.0, 22145, 65031, 53.2]
rmse_snav = [54.0, 257.3, 19673, 63498, 94.3]

# ── Chart 2: Normalised RMSE (ratio to DGSAR) ──
fig, ax = plt.subplots(figsize=(10, 5))
def normalize(vals, base):
    return [v / b for v, b in zip(vals, base)]

norm_fs    = [1.0] * 5
norm_prop  = normalize(rmse_prop, rmse_fs)
norm_ets   = normalize(rmse_ets, rmse_fs)
norm_theta = normalize(rmse_theta, rmse_fs)
norm_snav  = normalize(rmse_snav, rmse_fs)

bars2 = [
    (norm_fs, 'DGSAR', '#2563eb'),
    (norm_prop, 'Prophet', '#dc2626'),
    (norm_ets, 'ETS', '#16a34a'),
    (norm_theta, 'Theta', '#ca8a04'),
    (norm_snav, 'Seasonal Naive', '#7c3aed'),
]
for i, (vals, label, color) in enumerate(bars2):
    display_vals = [min(v, 6.0) for v in vals]
    rects = ax.bar(x + (i - 2) * w, display_vals, w, label=label, color=color, alpha=0.85)
    for j, (dv, rv) in enumerate(zip(display_vals, vals)):
        if rv > 6.0:
            ax.text(x[j] + (i - 2) * w, dv + 0.05, f'{rv:.1f}x',
                    ha='center', va='bottom', fontsize=7, rotation=90)

ax.set_ylabel('RMSE (normalised to DGSAR = 1.0)')
ax.set_title('Relative RMSE Comparison (lower is better)')
ax.set_xticks(x)
ax.set_xticklabels(series)
ax.axhline(y=1.0, color='#2563eb', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_ylim(0, 6.5)
ax.legend(loc='upper right', ncol=2)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig('c:/Users/Swastik/Desktop/FutureScope/rmse_comparison.png', bbox_inches='tight')
fig.savefig('c:/Users/Swastik/Desktop/FutureScope/rmse_comparison.pdf', bbox_inches='tight')
print('Saved rmse_comparison.png/pdf')
plt.close()

# ── Chart 3: Diagnostic pass rates (grouped bar) ──
diag_series = ['M4 Hourly', 'Daily (D1)', 'Bitcoin', 'COVID-19', 'Airline']
lb_pass  = [100, 75, 100, 75, 75]
acf_pass = [100, 100, 100, 100, 100]
arch_pass = [0, 0, 100, 100, 75]

fig, ax = plt.subplots(figsize=(9, 5))
x2 = np.arange(len(diag_series))
w2 = 0.25

ax.bar(x2 - w2, lb_pass, w2, label='Ljung-Box Pass %', color='#2563eb', alpha=0.85)
ax.bar(x2, acf_pass, w2, label='ACF Pass %', color='#16a34a', alpha=0.85)
ax.bar(x2 + w2, arch_pass, w2, label='ARCH-LM Fail %', color='#dc2626', alpha=0.85)

ax.set_ylabel('Rate (%)')
ax.set_title('Diagnostic Pass Rates After Gated Refitting')
ax.set_xticks(x2)
ax.set_xticklabels(diag_series)
ax.set_ylim(0, 115)
ax.legend()
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig('c:/Users/Swastik/Desktop/FutureScope/diagnostic_comparison.png', bbox_inches='tight')
fig.savefig('c:/Users/Swastik/Desktop/FutureScope/diagnostic_comparison.pdf', bbox_inches='tight')
print('Saved diagnostic_comparison.png/pdf')
plt.close()

# ── Chart 4: Win rate heatmap-style bar chart ──
baselines = ['Prophet', 'ETS', 'Theta', 'Seasonal\nNaive']
win_rmse = [3, 4, 3, 3]  # out of 5
win_mase = [5, 3, 3, 4]

fig, ax = plt.subplots(figsize=(7, 4.5))
x3 = np.arange(len(baselines))
w3 = 0.3

ax.bar(x3 - w3/2, win_rmse, w3, label='RMSE Win Rate', color='#2563eb', alpha=0.85)
ax.bar(x3 + w3/2, win_mase, w3, label='MASE Win Rate', color='#16a34a', alpha=0.85)

ax.set_ylabel('Wins (out of 5 series)')
ax.set_title('DGSAR Win Rates vs Baselines')
ax.set_xticks(x3)
ax.set_xticklabels(baselines)
ax.set_ylim(0, 5.8)
ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.axhline(y=2.5, color='gray', linestyle=':', linewidth=0.8, label='Parity (2.5/5)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig('c:/Users/Swastik/Desktop/FutureScope/winrate_comparison.png', bbox_inches='tight')
fig.savefig('c:/Users/Swastik/Desktop/FutureScope/winrate_comparison.pdf', bbox_inches='tight')
print('Saved winrate_comparison.png/pdf')
plt.close()

print('\nAll charts generated successfully.')
