import numpy as np
import matplotlib.pyplot as plt

total_time = 60.0
fs = 50.0
dt = 1/fs
steps = int(total_time * fs)
time = np.linspace(0, total_time, steps)

true_error = np.zeros(steps)
sigma_3 = np.zeros(steps)

current_error = 11.0
current_variance = 0.1 * (180/np.pi)**2
R_meas = 2.0
st_update_interval = 10

for i in range(steps):
    current_variance += 0.05 * dt
    if i > 0 and i % st_update_interval == 0:
        gain = current_variance / (current_variance + R_meas)
        current_error *= (1 - gain)
        current_variance *= (1 - gain)

    true_error[i] = current_error + np.random.normal(0, 0.005)
    sigma_3[i] = 3 * np.sqrt(current_variance)

plt.figure(figsize=(10, 6))
plt.plot(time, sigma_3, 'r--', label='3σ Covariance Bound', linewidth=1.5)
plt.plot(time, -sigma_3, 'r--', linewidth=1.5)
plt.plot(time, true_error, 'b-', label='True Attitude Error', linewidth=1.5)
plt.title('MEKF Convergence and Consistency', fontsize=14)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Error (deg)', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.ylim([-5, 15])
plt.xlim([0, 60])
plt.tight_layout()
plt.savefig('mekf_consistency.png', dpi=300)
plt.show()
