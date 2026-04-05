import matplotlib.pyplot as plt
import numpy as np

# --- Your Data ---
labels = ['Baseline Pilot', 'Robust Pilot']
crash_rates = [100.0, 0.0]        # From your adversarial validation
offsets = [0.0737, 0.2109]        # From your precision validation

# --- Setup Dark Mode Style ---
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# --- Chart 1: The Win (Crash Rate in Wind) ---
bars1 = ax1.bar(labels, crash_rates, color=['#ff4d4d', '#50E3C2'], width=0.5)
ax1.set_title('Adversarial Environment: Crash Rate', fontsize=14, fontweight='bold')
ax1.set_ylabel('Crash Percentage (%)', fontsize=12)
ax1.set_ylim(0, 110)

# Add data labels on top of bars
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# --- Chart 2: The Trade-off (Precision in Vacuum) ---
bars2 = ax2.bar(labels, offsets, color=['#4A90E2', '#f39c12'], width=0.5)
ax2.set_title('Clean Environment: Landing Precision', fontsize=14, fontweight='bold')
ax2.set_ylabel('Distance from Target (Lower is Better)', fontsize=12)
ax2.set_ylim(0, 0.3)

# Add data labels on top of bars
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# --- Final Polish ---
plt.suptitle('The Stability Tax: Fragile Precision vs. Absolute Robustness', fontsize=16, y=1.05)
plt.tight_layout()
plt.savefig('presentation_charts.png', bbox_inches='tight')
print("Charts generated successfully as 'presentation_charts.png'")