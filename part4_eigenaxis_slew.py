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

Kp = np.diag([0.015, 0.015, 0.020])
Kd = np.diag([0.12, 0.12, 0.15])

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

def dynamics_step(q, omega, wheel_h, tau_cmd, dt, theta_orbit):
    tau_env = gravity_gradient_torque(q, theta_orbit) + drag_torque()
    domega = I_inv @ (tau_cmd + tau_env - np.cross(omega, I_sc @ omega))
    omega_new = omega + domega * dt
    q_new = hw3.propagate_quat(q, omega_new, dt)
    wheel_h_new = wheel_h - tau_cmd * dt
    return q_new, omega_new, wheel_h_new, tau_env

def axis_angle_from_quat(q):
    q = hw3.q_normalize(q)
    if q[0] < 0:
        q = -q
    angle = 2.0 * np.arccos(np.clip(q[0], -1.0, 1.0))
    s = np.sqrt(max(1.0 - q[0]**2, 0.0))
    if s < 1e-12:
        return np.array([1.0, 0.0, 0.0]), 0.0
    axis = q[1:4] / s
    return axis, angle

def versine_profile(phi, T, t):
    theta = 0.5 * phi * (1.0 - np.cos(np.pi * t / T))
    theta_dot = 0.5 * phi * (np.pi / T) * np.sin(np.pi * t / T)
    theta_ddot = 0.5 * phi * (np.pi / T)**2 * np.cos(np.pi * t / T)
    return theta, theta_dot, theta_ddot

def desired_slew_trajectory(q0, qf, T, t):
    q_rel = hw3.q_mult(qf, hw3.q_conj(q0))
    axis, phi = axis_angle_from_quat(q_rel)

    t = np.clip(t, 0.0, T)
    theta, theta_dot, theta_ddot = versine_profile(phi, T, t)

    q_inc = hw3.rotvec_to_quat(axis * theta)
    qd = hw3.q_normalize(hw3.q_mult(q_inc, q0))
    omega_d = axis * theta_dot
    domega_d = axis * theta_ddot

    return qd, omega_d, domega_d

def tracking_control(qd, omega_d, domega_d, q_hat, omega_hat):
    dtheta = small_angle_error(qd, q_hat)
    tau_ff = I_sc @ domega_d + np.cross(omega_d, I_sc @ omega_d)
    tau_fb = -Kp @ dtheta - Kd @ (omega_hat - omega_d)
    tau = tau_ff + tau_fb
    return np.clip(tau, -tau_max, tau_max), tau_ff, tau_fb

def run_slew_case(Tslew=70.0, total_time=120.0, seed=5, show_progress=True):
    rng = np.random.default_rng(seed)

    dt = 0.02
    st_dt = 0.1
    st_interval = int(round(st_dt / dt))
    N = int(total_time / dt)

    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    qf = hw3.rotvec_to_quat(np.array([0.0, 0.0, np.pi]))

    q_true = q0.copy()
    omega_true = np.zeros(3)
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
    err_track_deg = np.zeros(N)
    err_final_deg = np.zeros(N)
    tau_hist = np.zeros((N,3))
    tauff_hist = np.zeros((N,3))
    taufb_hist = np.zeros((N,3))
    omega_hist = np.zeros((N,3))
    omega_d_hist = np.zeros((N,3))
    h_hist = np.zeros((N,3))

    iterator = range(N)
    if show_progress:
        iterator = tqdm(iterator, total=N, desc="Slew sim", unit="step")

    for k in iterator:
        t = time_arr[k]
        theta_orbit = omega_orbit * t

        omega_meas = hw3.simulate_gyro_measurement(omega_true, b_true, M_g, dt, rng)
        q_hat, b_hat, P = hw3.mekf_propagate(q_hat, b_hat, P, omega_meas, dt)

        if k % st_interval == 0:
            q_st = hw3.simulate_star_tracker_measurement(q_true, rng)
            q_hat, b_hat, P = hw3.mekf_update_star_tracker(q_hat, b_hat, P, q_st)

        omega_hat = omega_meas - b_hat
        qd, omega_d, domega_d = desired_slew_trajectory(q0, qf, Tslew, t)
        tau_cmd, tau_ff, tau_fb = tracking_control(qd, omega_d, domega_d, q_hat, omega_hat)

        q_true, omega_true, wheel_h, _ = dynamics_step(q_true, omega_true, wheel_h, tau_cmd, dt, theta_orbit)
        b_true = hw3.propagate_true_bias(b_true, dt, rng)

        err_track_deg[k] = hw3.quat_angle_error_deg(qd, q_true)
        err_final_deg[k] = hw3.quat_angle_error_deg(qf, q_true)
        tau_hist[k] = tau_cmd
        tauff_hist[k] = tau_ff
        taufb_hist[k] = tau_fb
        omega_hist[k] = np.rad2deg(omega_true)
        omega_d_hist[k] = np.rad2deg(omega_d)
        h_hist[k] = wheel_h

    peak_tau = np.max(np.abs(tau_hist))
    peak_h = np.max(np.abs(h_hist))
    final_err = err_final_deg[-1]

    post_slew_mask = time_arr >= Tslew
    post_slew_rms = np.sqrt(np.mean(err_final_deg[post_slew_mask]**2))

    fig, axs = plt.subplots(3, 1, figsize=(10,10), sharex=True)

    axs[0].plot(time_arr, err_track_deg, label='Tracking Error')
    axs[0].plot(time_arr, err_final_deg, label='Final Attitude Error')
    axs[0].axvline(Tslew, color='k', linestyle='--', alpha=0.7, label='Slew End')
    axs[0].set_ylabel('Error (deg)')
    axs[0].set_title('180° Eigen-Axis Slew')
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].legend()

    axs[1].plot(time_arr, omega_hist[:,0], label='wx')
    axs[1].plot(time_arr, omega_hist[:,1], label='wy')
    axs[1].plot(time_arr, omega_hist[:,2], label='wz')
    axs[1].plot(time_arr, omega_d_hist[:,0], '--', color='C0')
    axs[1].plot(time_arr, omega_d_hist[:,1], '--', color='C1')
    axs[1].plot(time_arr, omega_d_hist[:,2], '--', color='C2')
    axs[1].axvline(Tslew, color='k', linestyle='--', alpha=0.7)
    axs[1].set_ylabel('Rate (deg/s)')
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].legend()

    axs[2].plot(time_arr, tau_hist[:,0], label='Tx')
    axs[2].plot(time_arr, tau_hist[:,1], label='Ty')
    axs[2].plot(time_arr, tau_hist[:,2], label='Tz')
    axs[2].axhline(tau_max, color='r', linestyle='--')
    axs[2].axhline(-tau_max, color='r', linestyle='--')
    axs[2].axvline(Tslew, color='k', linestyle='--', alpha=0.7)
    axs[2].set_ylabel('Torque (Nm)')
    axs[2].set_xlabel('Time (s)')
    axs[2].grid(True, linestyle='--', alpha=0.5)
    axs[2].legend()

    plt.tight_layout()
    plt.savefig('part4_slew_states.png', dpi=300)
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(time_arr, tauff_hist[:,0], '--', label='FF x')
    plt.plot(time_arr, tauff_hist[:,1], '--', label='FF y')
    plt.plot(time_arr, tauff_hist[:,2], '--', label='FF z')
    plt.plot(time_arr, tau_hist[:,0], label='Cmd x')
    plt.plot(time_arr, tau_hist[:,1], label='Cmd y')
    plt.plot(time_arr, tau_hist[:,2], label='Cmd z')
    plt.axhline(tau_max, color='r', linestyle='--')
    plt.axhline(-tau_max, color='r', linestyle='--')
    plt.axvline(Tslew, color='k', linestyle='--', alpha=0.7, label='Slew End')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.title('Eigen-Axis Slew Control Inputs')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('part4_slew_inputs.png', dpi=300)
    plt.show()

    print(f"Slew time T = {Tslew:.2f} s")
    print(f"Total simulation time = {total_time:.2f} s")
    print(f"Post-slew RMS final-attitude error = {post_slew_rms:.6f} deg")
    print(f"Final attitude error = {final_err:.6f} deg")
    print(f"Peak commanded torque = {peak_tau:.6f} Nm")
    print(f"Peak wheel momentum = {peak_h:.6f} Nms")
    print(f"Torque limit = {tau_max:.6f} Nm")
    print(f"Momentum limit = {h_max:.6f} Nms")

if __name__ == "__main__":
    t0 = time.time()
    run_slew_case(Tslew=70.0, total_time=120.0, seed=5, show_progress=True)
    print(f"Elapsed wall time = {time.time() - t0:.2f} s")
