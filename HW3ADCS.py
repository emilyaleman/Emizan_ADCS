import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def quat_mult(q, r):
    return np.array([
        q[0]*r[0] - q[1]*r[1] - q[2]*r[2] - q[3]*r[3],
        q[0]*r[1] + q[1]*r[0] + q[2]*r[3] - q[3]*r[2],
        q[0]*r[2] - q[1]*r[3] + q[2]*r[0] + q[3]*r[1],
        q[0]*r[3] + q[1]*r[2] - q[2]*r[1] + q[3]*r[0]
    ])

def quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

# sim setup
dt = 0.02
T = 60
steps = int(T/dt)
J = np.diag([0.0867, 0.1817, 0.1117])
J_inv = np.linalg.inv(J)

# truth state
q_true = np.array([1.0, 0, 0, 0])
w_true = np.array([0.02, 0.03, 0.01]) # ~2-3 deg/s tumble
b_true = np.array([1e-4, -1e-4, 5e-5])

# estimator state
q_est = np.array([0.995, 0.1, 0, 0]) # initialized w/ 11 deg error
q_est /= np.linalg.norm(q_est)
b_est = np.zeros(3)
P = block_diag(np.eye(3)*(0.1**2), np.eye(3)*(1e-4**2))

# noise parameters
sigma_st = 2.91e-5
R_meas = np.eye(3) * (sigma_st**2)
Q = block_diag(np.eye(3)*(1e-3)**2, np.eye(3)*(1e-5)**2)
vI = np.array([1, 0, 0])

err_deg, sig_deg, t_arr = [], [], []
np.random.seed(42)

for k in range(steps):
    t = k * dt

    w_dot = J_inv @ (-np.cross(w_true, J @ w_true))
    w_true = w_true + w_dot * dt
    q_dot = 0.5 * quat_mult(q_true, np.append(0, w_true))
    q_true = q_true + q_dot * dt
    q_true /= np.linalg.norm(q_true)

    w_meas = w_true + b_true + np.random.normal(0, 1e-3, 3)
    Rb_true = np.array([
        [1-2*(q_true[2]**2+q_true[3]**2), 2*(q_true[1]*q_true[2]-q_true[3]*q_true[0]), 2*(q_true[1]*q_true[3]+q_true[2]*q_true[0])],
        [2*(q_true[1]*q_true[2]+q_true[3]*q_true[0]), 1-2*(q_true[1]**2+q_true[3]**2), 2*(q_true[2]*q_true[3]-q_true[1]*q_true[0])],
        [2*(q_true[1]*q_true[3]-q_true[2]*q_true[0]), 2*(q_true[2]*q_true[3]+q_true[1]*q_true[0]), 1-2*(q_true[1]**2+q_true[2]**2)]
    ])
    v_meas = Rb_true.T @ vI + np.random.multivariate_normal(np.zeros(3), R_meas)
    v_meas /= np.linalg.norm(v_meas)

    # estimator propagation
    w_est = w_meas - b_est
    q_dot_est = 0.5 * quat_mult(q_est, np.append(0, w_est))
    q_est = q_est + q_dot_est * dt
    q_est /= np.linalg.norm(q_est)

    F = np.block([[-skew(w_est), -np.eye(3)], [np.zeros((3,3)), np.zeros((3,3))]])
    P = P + (F @ P + P @ F.T + Q) * dt

    # update estimator
    Rb_est = np.array([
        [1-2*(q_est[2]**2+q_est[3]**2), 2*(q_est[1]*q_est[2]-q_est[3]*q_est[0]), 2*(q_est[1]*q_est[3]+q_est[2]*q_est[0])],
        [2*(q_est[1]*q_est[2]+q_est[3]*q_est[0]), 1-2*(q_est[1]**2+q_est[3]**2), 2*(q_est[2]*q_est[3]-q_est[1]*q_est[0])],
        [2*(q_est[1]*q_est[3]-q_est[2]*q_est[0]), 2*(q_est[2]*q_est[3]+q_est[1]*q_est[0]), 1-2*(q_est[1]**2+q_est[2]**2)]
    ])
    v_est = Rb_est.T @ vI

    H = np.block([skew(v_est), np.zeros((3,3))])
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R_meas)
    dx = K @ (v_meas - v_est)

    # corrections
    dq = np.append(1, 0.5 * dx[:3])
    q_est = quat_mult(q_est, dq)
    q_est /= np.linalg.norm(q_est)
    b_est += dx[3:]
    P = (np.eye(6) - K @ H) @ P

    # error
    q_err = quat_mult(quat_inv(q_true), q_est)
    err_angle = 2 * np.arccos(np.clip(q_err[0], -1.0, 1.0)) * (180/np.pi)
    sigma_bound = 3 * np.sqrt(np.max(np.diag(P[:3,:3]))) * (180/np.pi)

    err_deg.append(err_angle)
    sig_deg.append(sigma_bound)
    t_arr.append(t)

# plot
plt.figure(figsize=(8,4))
plt.plot(t_arr, err_deg, label="True Attitude Error", color='b')
plt.plot(t_arr, sig_deg, '--', label="3$\sigma$ Covariance Bound", color='r')
plt.title("MEKF Convergence and Consistency")
plt.xlabel("Time (s)"); plt.ylabel("Error (deg)")
plt.legend(); plt.grid(True)
plt.savefig("mekf_consistency.png")
