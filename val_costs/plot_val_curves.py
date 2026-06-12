import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))

def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    epochs = [int(r['epoch']) for r in rows]
    costs  = [float(r['mean_val_cost']) for r in rows]
    return np.array(epochs), np.array(costs)

seeds   = ['1234', '5678', '9012']
colors  = {'1234': '#1f77b4', '5678': '#ff7f0e', '9012': '#2ca02c'}
tm_data = {s: read_csv(os.path.join(BASE, f'val_cost_tempmlp_s{s}.csv'))  for s in seeds}
bl_data = {s: read_csv(os.path.join(BASE, f'val_cost_baseline_s{s}.csv')) for s in seeds}

fig = plt.figure(figsize=(16, 12))
fig.suptitle('Validation Cost Curves: Temp-MLP vs Baseline (3 seeds, 100 epochs)',
             fontsize=14, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.32)

# ── Row 0: per-seed curves ──────────────────────────────────────────────────
for col, s in enumerate(seeds):
    ax = fig.add_subplot(gs[0, col])
    ep_tm, c_tm = tm_data[s]
    ep_bl, c_bl = bl_data[s]

    ax.plot(ep_tm, c_tm, color=colors[s], lw=1.8, label='Temp-MLP')
    ax.plot(ep_bl, c_bl, color=colors[s], lw=1.8, linestyle='--', label='Baseline')

    # mark spike in Temp-MLP (any point > mean + 3*std)
    mean, std = c_tm.mean(), c_tm.std()
    spikes = ep_tm[c_tm > mean + 3 * std]
    for sp in spikes:
        ax.axvline(sp, color='red', lw=0.8, alpha=0.5, linestyle=':')

    ax.set_title(f'Seed {s}', fontsize=11)
    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel('Mean Val Cost', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

# ── Row 1: gap per seed + mean gap ─────────────────────────────────────────
for col, s in enumerate(seeds):
    ax = fig.add_subplot(gs[1, col])
    ep = tm_data[s][0]
    gap = bl_data[s][1] - tm_data[s][1]

    ax.plot(ep, gap, color=colors[s], lw=1.8)
    ax.axhline(0, color='black', lw=0.8, linestyle='--')
    ax.fill_between(ep, 0, gap, where=gap > 0, alpha=0.15, color=colors[s],
                    label='Temp-MLP better')
    ax.fill_between(ep, 0, gap, where=gap < 0, alpha=0.15, color='red',
                    label='Baseline better')
    ax.set_title(f'Gap (Baseline − Temp-MLP)  seed {s}', fontsize=10)
    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel('Gap', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

# ── Row 2 col 0-1: all curves overlaid ─────────────────────────────────────
ax_all = fig.add_subplot(gs[2, 0:2])
for s in seeds:
    ep_tm, c_tm = tm_data[s]
    ep_bl, c_bl = bl_data[s]
    ax_all.plot(ep_tm, c_tm, color=colors[s], lw=1.5,
                label=f'Temp-MLP s{s}')
    ax_all.plot(ep_bl, c_bl, color=colors[s], lw=1.5, linestyle='--',
                label=f'Baseline s{s}')

ax_all.set_title('All runs overlaid', fontsize=11)
ax_all.set_xlabel('Epoch', fontsize=9)
ax_all.set_ylabel('Mean Val Cost', fontsize=9)
ax_all.legend(fontsize=7, ncol=2)
ax_all.grid(True, alpha=0.3)
ax_all.tick_params(labelsize=8)

# ── Row 2 col 2: mean ± std band ────────────────────────────────────────────
ax_mean = fig.add_subplot(gs[2, 2])

ep_ref = tm_data[seeds[0]][0]
tm_stack = np.stack([tm_data[s][1] for s in seeds])
bl_stack = np.stack([bl_data[s][1] for s in seeds])

tm_mean = tm_stack.mean(0); tm_std = tm_stack.std(0)
bl_mean = bl_stack.mean(0); bl_std = bl_stack.std(0)

ax_mean.plot(ep_ref, tm_mean, color='steelblue', lw=2, label='Temp-MLP mean')
ax_mean.fill_between(ep_ref, tm_mean - tm_std, tm_mean + tm_std,
                     alpha=0.2, color='steelblue')

ax_mean.plot(ep_ref, bl_mean, color='darkorange', lw=2,
             linestyle='--', label='Baseline mean')
ax_mean.fill_between(ep_ref, bl_mean - bl_std, bl_mean + bl_std,
                     alpha=0.2, color='darkorange')

ax_mean.set_title('Mean ± std across 3 seeds', fontsize=11)
ax_mean.set_xlabel('Epoch', fontsize=9)
ax_mean.set_ylabel('Mean Val Cost', fontsize=9)
ax_mean.legend(fontsize=8)
ax_mean.grid(True, alpha=0.3)
ax_mean.tick_params(labelsize=8)

out = os.path.join(BASE, 'val_cost_curves.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
plt.show()
