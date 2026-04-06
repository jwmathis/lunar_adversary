import matplotlib.pyplot as plt
import numpy as np

# --- Your Data ---
labels = ['Baseline Pilot', 'Robust Pilot']
crash_rates = [100.0, 0.0]        
offsets = [0.0737, 0.2109]        

# --- Setup Light Mode / Academic Style ---
plt.style.use('default') # Removed dark mode
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.patch.set_facecolor('white') # Force white background

# --- Chart 1: The Win (Crash Rate in Wind) ---
# Using standard high-contrast Red and Blue
bars1 = ax1.bar(labels, crash_rates, color=['#d9534f', '#5bc0de'], edgecolor='black', width=0.5)
ax1.set_title('Adversarial Environment: Crash Rate', fontsize=14, fontweight='bold')
ax1.set_ylabel('Crash Percentage (%)', fontsize=12)
ax1.set_ylim(0, 110)
ax1.grid(axis='y', linestyle='--', alpha=0.7) # Added grid for readability

# Add data labels on top of bars
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# --- Chart 2: The Trade-off (Precision in Vacuum) ---
bars2 = ax2.bar(labels, offsets, color=['#5bc0de', '#f0ad4e'], edgecolor='black', width=0.5)
ax2.set_title('Clean Environment: Landing Precision', fontsize=14, fontweight='bold')
ax2.set_ylabel('Distance from Target (Lower is Better)', fontsize=12)
ax2.set_ylim(0, 0.3)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Add data labels on top of bars
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# --- Final Polish ---
plt.suptitle('The Stability Tax: Fragile Precision vs. Absolute Robustness', fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('paper_charts_light.png', bbox_inches='tight', dpi=300) # Added DPI for print quality
print("Academic charts generated successfully as 'paper_charts_light.png'")