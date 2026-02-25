import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 1. Inertia Matrix (Diagonal Entries from Part 3)

J1 = 0.0867  # Minor
J2 = 0.1817  # Major
J3 = 0.1117  # Intermediate


# 2. Euler Equations (Torque-Free)

def euler_equations(t, omega):
    w1, w2, w3 = omega

    dw1 = ((J2 - J3) * w2 * w3) / J1
    dw2 = ((J3 - J1) * w3 * w1) / J2
    dw3 = ((J1 - J2) * w1 * w2) / J3

    return [dw1, dw2, dw3]


# 3. Simulation Parameters

t_span = (0, 50)
t_eval = np.linspace(*t_span, 2000)

rpm_to_rads = (2 * np.pi) / 60
base_speed = 10 * rpm_to_rads
perturbation = 0.01

scenarios = {
    "Major Axis (Stable)": [perturbation, base_speed, perturbation],
    "Minor Axis (Stable)": [base_speed, perturbation, perturbation],
    "Intermediate Axis (Unstable)": [perturbation, perturbation, base_speed]
}


# 4. Time History Plots

plt.figure(figsize=(12, 8))

for i, (name, w0) in enumerate(scenarios.items()):
    sol = solve_ivp(
        euler_equations,
        t_span,
        w0,
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12
    )

    plt.subplot(3, 1, i+1)
    plt.plot(sol.t, sol.y[0], label=r'$\omega_1$ (Minor)')
    plt.plot(sol.t, sol.y[1], label=r'$\omega_2$ (Major)')
    plt.plot(sol.t, sol.y[2], label=r'$\omega_3$ (Inter.)')
    plt.title(name)
    plt.ylabel("rad/s")
    plt.grid(True)
    plt.legend()

plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()

# 5. Momentum Sphere Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Sphere mesh
u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
sx = np.cos(u) * np.sin(v)
sy = np.sin(u) * np.sin(v)
sz = np.cos(v)

for name, w0 in scenarios.items():

    sol = solve_ivp(
        euler_equations,
        t_span,
        w0,
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12
    )

    # Angular momentum in body frame
    h1 = J1 * sol.y[0]
    h2 = J2 * sol.y[1]
    h3 = J3 * sol.y[2]

    # Compute constant magnitude from initial condition
    H0 = np.sqrt((J1*w0[0])**2 + (J2*w0[1])**2 + (J3*w0[2])**2)

    # Plot true momentum sphere
    ax.plot_wireframe(
        H0*sx,
        H0*sy,
        H0*sz,
        color='lightgrey',
        alpha=0.25,
        linewidth=0.5
    )

    # Plot trajectory 
    ax.plot(h1, h2, h3, linewidth=2.5, label=name)

    # checking conservation
    H_mag = np.sqrt(h1**2 + h2**2 + h3**2)
    print(f"{name} | Std Dev of |H|:", np.std(H_mag))

# Axis formatting
ax.set_xlabel("H1 (Minor)")
ax.set_ylabel("H2 (Major)")
ax.set_zlabel("H3 (Intermediate)")
ax.set_title("Momentum Sphere (True Polhodes)")
ax.legend()

ax.view_init(elev=30, azim=45)
ax.set_box_aspect([1,1,1])

plt.show()

# Axis formatting
ax.set_xlabel("H1 (Minor)")
ax.set_ylabel("H2 (Major)")
ax.set_zlabel("H3 (Intermediate)")
ax.set_title("Momentum Sphere (True Polhodes)")
ax.legend()

ax.view_init(elev=30, azim=45)
ax.set_box_aspect([1,1,1])

plt.show()