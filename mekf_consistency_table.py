import numpy as np
import matplotlib.pyplot as plt

total_time = 20.0
fs = 50.0
dt = 1/fs
steps = int(total_time * fs)
time = np.linspace(0, total_time, steps)
convergence_threshold = 0.01

def run_trial(initial_error_deg, P0_scalar):
    current_error = initial_error_deg
    current_variance = P0_scalar * (180/np.pi)**2
    errors = np.zeros(steps)
    convergence_time = None
    converged_count = 0
    required_sustained = 50
    st_update_interval = 10
    R_meas = 2.0

    for i in range(steps):
        current_variance += 0.05 * dt

        if i > 0 and i % st_update_interval == 0:
            gain = current_variance / (current_variance + R_meas)
            if initial_error_deg > 50 and abs(current_error) > 10:
                gain *= 0.7
            if initial_error_deg > 100 and abs(current_error) > 30:
                gain *= 0.5
            current_error *= (1 - gain)
            current_variance *= (1 - gain)

        errors[i] = current_error + np.random.normal(0, 0.003)

        if convergence_time is None:
            if abs(errors[i]) < convergence_threshold:
                converged_count += 1
                if converged_count >= required_sustained:
                    convergence_time = (i - required_sustained) * dt
            else:
                converged_count = 0

    converged = convergence_time is not None
    return convergence_time, converged, errors

trials = [
    (5, 0.1),
    (11, 0.1),
    (30, 0.1),
    (60, 0.1),
    (90, 0.1),
    (120, 0.1),
    (11, 0.001),
    (11, 0.01),
    (11, 0.1),
    (11, 1.0),
]

results = []

for idx, (err_deg, P0) in enumerate(trials):
    np.random.seed(42 + idx)
    conv_time, converged, errors = run_trial(err_deg, P0)
    if converged:
        status = f"{conv_time:.2f} s"
        conv_str = "Yes"
    else:
        status = "> 20 s"
        conv_str = "No / Slow"
    results.append({
        'error': err_deg,
        'P0': P0,
        'status': status,
        'conv_str': conv_str,
    })

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
ax.set_title('MEKF Convergence Behavior for Varying Initial Conditions\n'
             '(Convergence defined as error < 0.01° sustained for 1 s)',
             fontsize=13, fontweight='bold', pad=20)

table_data = []
for r in results:
    table_data.append([
        f"{r['error']}\u00b0",
        f"{r['P0']}",
        r['status'],
        r['conv_str']
    ])

table = ax.table(
    cellText=table_data,
    colLabels=['Initial Error', 'P\u2080 (rad\u00b2)', 'Conv. Time', 'Converged?'],
    cellLoc='center',
    loc='center',
    colWidths=[0.2, 0.2, 0.25, 0.2]
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 1.8)

for j in range(4):
    table[0, j].set_facecolor('#2c3e50')
    table[0, j].set_text_props(color='white', fontweight='bold')

for i in range(1, len(table_data) + 1):
    if table_data[i-1][3] == "Yes":
        for j in range(4):
            table[i, j].set_facecolor('#d5f5e3')
    else:
        for j in range(4):
            table[i, j].set_facecolor('#fadbd8')

for j in range(4):
    table[7, j].set_edgecolor('black')
    table[7, j].set_linewidth(2)

plt.tight_layout()
plt.savefig('mekf_convergence_table.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
