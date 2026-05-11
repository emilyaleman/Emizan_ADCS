import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def skew(v):
    return np.array([
        [0.0, -v[2],  v[1]],
        [v[2],  0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def q_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def q_normalize(q):
    return q / np.linalg.norm(q)

def rotvec_to_quat(phi):
    angle = np.linalg.norm(phi)
    if angle < 1e-12:
        return q_normalize(np.array([1.0, 0.5*phi[0], 0.5*phi[1], 0.5*phi[2]]))
    axis = phi / angle
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), axis[0]*s, axis[1]*s, axis[2]*s])

def propagate_quat(q, omega_body, dt):
    dq = rotvec_to_quat(omega_body * dt)
    return q_normalize(q_mult(q, dq))

def quat_error_small_angle(q_true, q_est):
    q_err = q_mult(q_true, q_conj(q_est))
    q_err = q_normalize(q_err)
    if q_err[0] < 0:
        q_err = -q_err
    return 2.0 * q_err[1:4]

def quat_angle_error_deg(q_true, q_est):
    q_err = q_mult(q_true, q_conj(q_est))
    q_err = q_normalize(q_err)
    ang = 2.0 * np.arccos(np.clip(abs(q_err[0]), -1.0, 1.0))
    return np.rad2deg(ang)

sigma_t = 5.236e-5
sigma_b = 1.222e-4
R_st = np.diag([sigma_t**2, sigma_t**2, sigma_b**2])

scale_sigma = 500e-6
misalign_sigma = 1e-3
sigma_arw = 0.15 * np.pi / 180.0 / 60.0
sigma_bg_rw = 0.3 * np.pi / 180.0 / 3600.0

def sample_gyro_calibration_matrix(rng):
    M = np.eye(3)
    diag_err = rng.normal(0.0, scale_sigma, size=3)
    M[np.diag_indices(3)] += diag_err
    offdiag = rng.normal(0.0, misalign_sigma, size=(3,3))
    offdiag[np.diag_indices(3)] = 0.0
    M += offdiag
    return M

def simulate_star_tracker_measurement(q_true, rng):
    dtheta = rng.multivariate_normal(np.zeros(3), R_st)
    dq = rotvec_to_quat(dtheta)
    return q_normalize(q_mult(dq, q_true))

def propagate_true_bias(b_true, dt, rng):
    return b_true + sigma_bg_rw * np.sqrt(dt) * rng.normal(size=3)

def simulate_gyro_measurement(omega_true, b_true, M_g, dt, rng):
    white_noise = (sigma_arw / np.sqrt(dt)) * rng.normal(size=3)
    return M_g @ omega_true + b_true + white_noise

def mekf_propagate(q_hat, b_hat, P, omega_meas, dt):
    omega_hat = omega_meas - b_hat
    q_hat_new = propagate_quat(q_hat, omega_hat, dt)
    b_hat_new = b_hat.copy()

    F = np.block([
        [-skew(omega_hat), -np.eye(3)],
        [np.zeros((3,3)),  np.zeros((3,3))]
    ])

    G = np.block([
        [-np.eye(3), np.zeros((3,3))],
        [np.zeros((3,3)), np.eye(3)]
    ])

    Qc = np.block([
        [sigma_arw**2 * np.eye(3), np.zeros((3,3))],
        [np.zeros((3,3)), sigma_bg_rw**2 * np.eye(3)]
    ])

    Phi = np.eye(6) + F * dt
    Qd = G @ Qc @ G.T * dt

    P_new = Phi @ P @ Phi.T + Qd
    P_new = 0.5 * (P_new + P_new.T)

    return q_hat_new, b_hat_new, P_new

def mekf_update_star_tracker(q_hat, b_hat, P, q_meas):
    q_tilde = q_mult(q_meas, q_conj(q_hat))
    q_tilde = q_normalize(q_tilde)
    if q_tilde[0] < 0:
        q_tilde = -q_tilde

    z = 2.0 * q_tilde[1:4]
    H = np.hstack((np.eye(3), np.zeros((3,3))))
    S = H @ P @ H.T + R_st
    K = P @ H.T @ np.linalg.inv(S)

    dx = K @ z
    dtheta = dx[:3]
    dbias = dx[3:]

    dq = rotvec_to_quat(dtheta)
    q_hat_new = q_normalize(q_mult(dq, q_hat))
    b_hat_new = b_hat + dbias

    I6 = np.eye(6)
    P_new = (I6 - K @ H) @ P @ (I6 - K @ H).T + K @ R_st @ K.T
    P_new = 0.5 * (P_new + P_new.T)

    return q_hat_new, b_hat_new, P_new

def run_trial(
    total_time=60.0,
    gyro_rate_hz=50.0,
    star_tracker_rate_hz=5.0,
    initial_error_deg=11.0,
    P0_att=0.1,
    P0_bias=0.01,
    seed=1
):
    rng = np.random.default_rng(seed)

    dt = 1.0 / gyro_rate_hz
    N = int(total_time * gyro_rate_hz)
    st_interval = int(round(gyro_rate_hz / star_tracker_rate_hz))

    omega_true = np.deg2rad(np.array([1.5, -2.0, 1.0]))

    q_true = np.array([1.0, 0.0, 0.0, 0.0])
    b_true = np.zeros(3)
    M_g = sample_gyro_calibration_matrix(rng)

    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    phi0 = np.deg2rad(initial_error_deg) * axis
    q_hat = q_normalize(q_mult(rotvec_to_quat(phi0), q_true))
    b_hat = np.zeros(3)

    P = np.block([
        [P0_att * np.eye(3), np.zeros((3,3))],
        [np.zeros((3,3)), P0_bias * np.eye(3)]
    ])

    time = np.arange(N) * dt
    err_deg_hist = np.zeros(N)
    att_err_rad_hist = np.zeros((N,3))
    sigma3_deg_hist = np.zeros((N,3))
    static_err_deg_hist = np.full(N, np.nan)

    for k in range(N):
        dtheta_true = quat_error_small_angle(q_true, q_hat)
        att_err_rad_hist[k] = dtheta_true
        err_deg_hist[k] = quat_angle_error_deg(q_true, q_hat)
        sigma3_deg_hist[k] = np.rad2deg(3.0 * np.sqrt(np.diag(P[:3,:3])))

        q_true = propagate_quat(q_true, omega_true, dt)
        b_true = propagate_true_bias(b_true, dt, rng)

        omega_meas = simulate_gyro_measurement(omega_true, b_true, M_g, dt, rng)
        q_hat, b_hat, P = mekf_propagate(q_hat, b_hat, P, omega_meas, dt)

        if (k + 1) % st_interval == 0:
            q_st = simulate_star_tracker_measurement(q_true, rng)
            static_err_deg_hist[k] = quat_angle_error_deg(q_true, q_st)
            q_hat, b_hat, P = mekf_update_star_tracker(q_hat, b_hat, P, q_st)

    return {
        "time": time,
        "err_deg": err_deg_hist,
        "att_err_rad": att_err_rad_hist,
        "sigma3_deg": sigma3_deg_hist,
        "static_err_deg": static_err_deg_hist
    }

def convergence_time(err_deg, dt, threshold_deg=0.01, sustain_sec=1.0):
    needed = int(round(sustain_sec / dt))
    count = 0
    for i, e in enumerate(err_deg):
        if e < threshold_deg:
            count += 1
            if count >= needed:
                return (i - needed + 1) * dt
        else:
            count = 0
    return None

def compute_3sigma_coverage(result):
    err = result["att_err_rad"]
    sigma3 = np.deg2rad(result["sigma3_deg"])
    inside = np.abs(err) <= sigma3
    axis_coverage = 100.0 * np.mean(inside, axis=0)
    joint_coverage = 100.0 * np.mean(np.all(inside, axis=1))
    return axis_coverage, joint_coverage

def monte_carlo_summary(initial_error_deg, P0_att, n_runs=30):
    conv_times = []
    successes = 0

    for k in range(n_runs):
        res = run_trial(
            total_time=20.0,
            gyro_rate_hz=50.0,
            star_tracker_rate_hz=5.0,
            initial_error_deg=initial_error_deg,
            P0_att=P0_att,
            P0_bias=0.01,
            seed=1000 + 100*k + int(10*initial_error_deg)
        )
        tconv = convergence_time(
            res["err_deg"],
            dt=1/50.0,
            threshold_deg=0.01,
            sustain_sec=1.0
        )
        if tconv is not None:
            successes += 1
            conv_times.append(tconv)

    if len(conv_times) == 0:
        return None, successes / n_runs
    return float(np.median(conv_times)), successes / n_runs

def compute_fair_static_vs_mekf_rms(result, settle_time=10.0):
    time = result["time"]
    static_err = result["static_err_deg"]
    mekf_err = result["err_deg"]

    mask = (~np.isnan(static_err)) & (time >= settle_time)

    static_rms = np.sqrt(np.mean(static_err[mask]**2))
    mekf_rms = np.sqrt(np.mean(mekf_err[mask]**2))

    return static_rms, mekf_rms

if __name__ == "__main__":
    result = run_trial(
        total_time=60.0,
        gyro_rate_hz=50.0,
        star_tracker_rate_hz=5.0,
        initial_error_deg=11.0,
        P0_att=0.1,
        P0_bias=0.01,
        seed=10
    )

    time = result["time"]
    err_deg = result["err_deg"]
    sigma3_deg = result["sigma3_deg"]

    plt.figure(figsize=(10,6))
    plt.plot(time, err_deg, 'b', linewidth=1.5, label='True Attitude Error')
    plt.plot(time, sigma3_deg[:,0], 'r--', linewidth=1.5, label='3σ Bound')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Attitude Error (deg)')
    plt.title('MEKF Convergence and Consistency')
    plt.legend()
    plt.xlim(0, 15)
    plt.ylim(-0.2, 0.2)
    plt.tight_layout()
    plt.savefig('mekf_consistency.png', dpi=300)
    plt.show()

    axis_cov, joint_cov = compute_3sigma_coverage(result)
    print(f"3σ coverage axis 1: {axis_cov[0]:.2f}%")
    print(f"3σ coverage axis 2: {axis_cov[1]:.2f}%")
    print(f"3σ coverage axis 3: {axis_cov[2]:.2f}%")
    print(f"3σ coverage all axes simultaneously: {joint_cov:.2f}%")

    static_rms, mekf_rms = compute_fair_static_vs_mekf_rms(result, settle_time=10.0)
    print(f"Static star-tracker-only RMS error after 10 s: {static_rms:.6f} deg")
    print(f"MEKF RMS error after 10 s: {mekf_rms:.6f} deg")

    trials = [
        (5,   0.1),
        (11,  0.1),
        (30,  0.1),
        (60,  0.1),
        (90,  0.1),
        (120, 0.1),
        (11,  0.001),
        (11,  0.01),
        (11,  0.1),
        (11,  1.0),
    ]

    rows = []
    for init_err, P0_att in trials:
        med_t, success_rate = monte_carlo_summary(init_err, P0_att, n_runs=30)
        rows.append([
            f"{init_err}°",
            f"{P0_att}",
            ">20 s" if med_t is None else f"{med_t:.2f} s",
            f"{100*success_rate:.0f}%"
        ])

    df = pd.DataFrame(rows, columns=[
        "Initial Error",
        "P₀ (rad²)",
        "Median Conv. Time",
        "Success Rate"
    ])

    print(df)

    fig, ax = plt.subplots(figsize=(10,6))
    ax.axis('off')
    ax.set_title(
        'MEKF Convergence Behavior for Varying Initial Conditions\n'
        '(Convergence: error < 0.01° sustained for 1 s)',
        fontsize=13,
        fontweight='bold',
        pad=20
    )

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.20, 0.18, 0.28, 0.20]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    for j in range(len(df.columns)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    for i in range(1, len(df) + 1):
        success_val = int(df.iloc[i-1, 3].replace('%', ''))
        color = '#d5f5e3' if success_val == 100 else '#fadbd8'
        for j in range(len(df.columns)):
            table[i, j].set_facecolor(color)

    plt.tight_layout()
    plt.savefig('mekf_convergence_table.png', dpi=300, bbox_inches='tight')
    plt.show()
