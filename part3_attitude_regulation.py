
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "hw3"))
import HW3ADCS as hw3

I_sc = np.diag([0.0867, 0.1817, 0.1117])
I_inv = np.linalg.inv(I_sc)

tau_max = 0.005
h_max = 0.015

mu = 3.986004418e14
Re = 6378e3
alt = 400e3
r_orbit = Re + alt
omega_orbit = np.sqrt(mu / r_orbit**3)

rho = 1e-12
Cd = 2.2
A_proj = 0.04
v_orbit = np.sqrt(mu / r_orbit)
r_cp = np.array([0.02, 0.0, 0.0])

Kp = np.diag([0.006, 0.006, 0.006])
Kd = np.diag([0.08, 0.08, 0.08])

def quat_to_rotmat(q):
    q = hw3.q_normalize(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])

def small_angle_error(q_des, q_est):
    q_err = hw3.q_mult(hw3.q_conj(q_des), q_est)
    q_err = hw3.q_normalize(q_err)
    if q_err[0] < 0:
        q_err = -q_err
    return 2.0 * q_err[1:4]

def gravity_gradient_torque(q, theta_orbit):
    R_bn = quat_to_rotmat(q)
    rhat_n = np.array([np.cos(theta_orbit), np.sin(theta_orbit), 0.0])
    rhat_b = R_bn.T @ rhat_n
    return 3.0 * omega_orbit**2 * np.cross(rhat_b, I_sc @ rhat_b)

def drag_torque():
    F_drag_mag = 0.5 * rho * Cd * A_proj * v_orbit**2
    F_drag_b = np.array([-F_drag_mag, 0.0, 0.0])
    return np.cross(r_cp, F_drag_b)

def control_torque(q_des, q_hat, omega_hat):
    dtheta = small_angle_error(q_des, q_hat)
    tau = -Kp @ dtheta - Kd @ omega_hat
    return np.clip(tau, -tau_max, tau_max)

def dynamics_step(q, omega, wheel_h, tau_cmd, dt, theta_orbit):
    tau_env = gravity_gradient_torque(q, theta_orbit) + drag_torque()
    domega = I_inv @ (tau_cmd + tau_env - np.cross(omega, I_sc @ omega))
    omega_new = omega + domega * dt
    q_new = hw3.propagate_quat(q, omega_new, dt)
    wheel_h_new = wheel_h - tau_cmd * dt
    return q_new, omega_new, wheel_h_new, tau_env

def random_quaternion_error(max_angle_deg, rng):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(rng.uniform(0.0, max_angle_deg))
    return hw3.rotvec_to_quat(axis * angle)

def compute_metrics(time_arr, err_deg, settle_time=100.0):
    full_rms = np.sqrt(np.mean(err_deg**2))
    final_err = err_deg[-1]

    steady_mask = time_arr >= settle_time
    if np.any(steady_mask):
        steady_rms = np.sqrt(np.mean(err_deg[steady_mask]**2))
    else:
        steady_rms = np.nan

    return full_rms, steady_rms, final_err

def run_single_trial(seed=1, total_time=300.0, q_des=None, show_progress=False, settle_time=100.0):
    if q_des is None:
        q_des = np.array([1.0, 0.0, 0.0, 0.0])

    rng = np.random.default_rng(seed)

    dt = 0.02
    st_dt = 0.2
    st_interval = int(round(st_dt / dt))
    N = int(total_time / dt)

    q_true = hw3.q_normalize(hw3.q_mult(random_quaternion_error(90.0, rng), q_des))
    omega_true = np.deg2rad(rng.uniform(-0.5, 0.5, size=3))
    b_true = np.zeros(3)
    M_g = hw3.sample_gyro_calibration_matrix(rng)
    wheel_h = np.zeros(3)

    q_hat = q_true.copy()
    b_hat = np.zeros(3)
    P = np.block([
        [0.1*np.eye(3), np.zeros((3,3))],
        [np.zeros((3,3)), 0.01*np.eye(3)]
    ])

    time_arr = np.arange(N) * dt
    err_deg = np.zeros(N)
    torque_hist = np.zeros((N,3))
    h_hist = np.zeros((N,3))
    env_hist = np.zeros((N,3))

    iterator = range(N)
    if show_progress:
        iterator = tqdm(iterator, total=N, desc="Simulating", unit="step")

    for k in iterator:
        theta_orbit = omega_orbit * time_arr[k]

        omega_meas = hw3.simulate_gyro_measurement(omega_true, b_true, M_g, dt, rng)
        q_hat, b_hat, P = hw3.mekf_propagate(q_hat, b_hat, P, omega_meas, dt)

        if k % st_interval == 0:
            q_st = hw3.simulate_star_tracker_measurement(q_true, rng)
            q_hat, b_hat, P = hw3.mekf_update_star_tracker(q_hat, b_hat, P, q_st)

        omega_hat = omega_meas - b_hat
        tau_cmd = control_torque(q_des, q_hat, omega_hat)

        q_true, omega_true, wheel_h, tau_env = dynamics_step(
            q_true, omega_true, wheel_h, tau_cmd, dt, theta_orbit
        )
        b_true = hw3.propagate_true_bias(b_true, dt, rng)

        err_deg[k] = hw3.quat_angle_error_deg(q_des, q_true)
        torque_hist[k] = tau_cmd
        h_hist[k] = wheel_h
        env_hist[k] = tau_env

    full_rms, steady_rms, final_err = compute_metrics(time_arr, err_deg, settle_time=settle_time)

    peak_torque = np.max(np.abs(torque_hist))
    peak_h = np.max(np.abs(h_hist))
    peak_env = np.max(np.linalg.norm(env_hist, axis=1))

    return {
        "time": time_arr,
        "err_deg": err_deg,
        "torque_hist": torque_hist,
        "h_hist": h_hist,
        "env_hist": env_hist,
        "full_rms_deg": full_rms,
        "steady_rms_deg": steady_rms,
        "final_err_deg": final_err,
        "peak_torque": peak_torque,
        "peak_h": peak_h,
        "peak_env": peak_env
    }

def run_random_trials(ntrials=5):
    plt.figure(figsize=(10,6))
    full_rms_list = []
    final_err_list = []

    for i in range(ntrials):
        out = run_single_trial(seed=100+i, total_time=200.0, settle_time=100.0)
        plt.plot(out["time"], out["err_deg"], linewidth=1.3, label=f'Trial {i+1}')
        full_rms_list.append(out["full_rms_deg"])
        final_err_list.append(out["final_err_deg"])

        print(
            f"Trial {i+1}: "
            f"full RMS = {out['full_rms_deg']:.6f} deg, "
            f"steady RMS = {out['steady_rms_deg']:.6f} deg, "
            f"final error = {out['final_err_deg']:.6f} deg, "
            f"peak torque = {out['peak_torque']:.6f} Nm, "
            f"peak wheel momentum = {out['peak_h']:.6f} Nms"
        )

    plt.xlabel('Time (s)')
    plt.ylabel('Pointing Error (deg)')
    plt.title('Attitude Regulation from Random Initial Conditions')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('part3_random_trials.png', dpi=300)
    plt.show()

    print(f"\nAverage full-run RMS error: {np.mean(full_rms_list):.6f} deg")
    print(f"Average final error: {np.mean(final_err_list):.6f} deg")

def run_multi_orbit_case():
    orbit_period = 2*np.pi / omega_orbit
    total_time = 3 * orbit_period

    print("Starting multi-orbit case...")
    print(f"Orbit period = {orbit_period/60:.2f} min")
    print(f"Total simulation time = {total_time/3600:.2f} hr")

    t0 = time.time()
    out = run_single_trial(seed=999, total_time=total_time, show_progress=True, settle_time=1000.0)
    elapsed = time.time() - t0

    time_hr = out["time"] / 3600.0

    fig, axs = plt.subplots(3, 1, figsize=(10,10), sharex=True)

    axs[0].plot(time_hr, out["err_deg"], 'b')
    axs[0].set_ylabel('Error (deg)')
    axs[0].set_title('Multi-Orbit Attitude Regulation')
    axs[0].grid(True, linestyle='--', alpha=0.5)

    axs[1].plot(time_hr, out["torque_hist"][:,0], label='Tx')
    axs[1].plot(time_hr, out["torque_hist"][:,1], label='Ty')
    axs[1].plot(time_hr, out["torque_hist"][:,2], label='Tz')
    axs[1].set_ylabel('Torque (Nm)')
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].legend()

    axs[2].plot(time_hr, out["h_hist"][:,0], label='Hx')
    axs[2].plot(time_hr, out["h_hist"][:,1], label='Hy')
    axs[2].plot(time_hr, out["h_hist"][:,2], label='Hz')
    axs[2].axhline(h_max, color='r', linestyle='--')
    axs[2].axhline(-h_max, color='r', linestyle='--')
    axs[2].set_ylabel('Wheel Momentum (Nms)')
    axs[2].set_xlabel('Time (hr)')
    axs[2].grid(True, linestyle='--', alpha=0.5)
    axs[2].legend()

    print(f"Simulation finished in {elapsed:.2f} s")
    print(f"Multi-orbit full-run RMS pointing error = {out['full_rms_deg']:.6f} deg")
    print(f"Multi-orbit steady-state RMS pointing error = {out['steady_rms_deg']:.6f} deg")
    print(f"Final pointing error = {out['final_err_deg']:.6f} deg")
    print(f"Peak commanded torque = {out['peak_torque']:.6f} Nm")
    print(f"Peak wheel momentum = {out['peak_h']:.6f} Nms")
    print(f"Peak environmental torque = {out['peak_env']:.6e} Nm")
    print(f"Wheel momentum limit = {h_max:.6f} Nms")

    plt.tight_layout()
    plt.savefig('part3_multi_orbit.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_random_trials(ntrials=5)
    run_multi_orbit_case()
