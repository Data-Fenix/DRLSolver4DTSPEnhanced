import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Bin boundaries (13 points for 12 bins of 2 hours each)
bin_hours = np.arange(0, 25, 2)

# Synthetic but realistic travel times (minutes) for one Beijing edge
# Two rush-hour peaks: morning ~08:00, evening ~18:00; low at night
travel_times = np.array([18, 14, 16, 28, 48, 35, 28, 30, 42, 52, 38, 24, 18])

# Fit C2-smooth cubic spline
cs = CubicSpline(bin_hours, travel_times, bc_type='not-a-knot')

# Dense curve for smooth plotting
t_fine = np.linspace(0, 24, 600)
tt_fine = cs(t_fine)

fig, ax = plt.subplots(figsize=(8, 3.8))
fig.patch.set_facecolor('#FFF9E6')
ax.set_facecolor('#FFF9E6')

# Bin boundary lines
for h in range(0, 25, 2):
    ax.axvline(x=h, color='#aaaaaa', linestyle='--', linewidth=0.7, alpha=0.6)

# Smooth spline curve
ax.plot(t_fine, tt_fine, color='#3a6b8a', linewidth=2.5, label='Travel time  $d_{ij}(t)$')

# Knot points
ax.scatter(bin_hours, travel_times, color='#2c2c2c', zorder=5, s=35, label='Bin boundary values')

# Axis formatting
ax.set_xlim(0, 24)
ax.set_ylim(0, 65)
ax.set_xticks(range(0, 25, 2))
ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 2)],
                   rotation=45, ha='right', fontsize=8.5)
ax.set_xlabel('Departure time', fontsize=10)
ax.set_ylabel('Travel time (min)', fontsize=10)
ax.set_title('Cubic Spline Travel Time Representation — single edge $i \\rightarrow j$',
             fontsize=11, fontweight='bold', pad=10)

ax.legend(fontsize=8.5, loc='upper left', framealpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('C:/Users/lahir/Downloads/Thesis/DRLSolver4DTSP-main/spline_plot.png',
            dpi=180, bbox_inches='tight')
print("Saved: spline_plot.png")
plt.show()
