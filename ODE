import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
mu = 398600.4418  # km^3/s^2
R_Earth = 6378.137  # km
alt = 400.0  # ASTERIA/ISS altitude in km

def orbital_dynamics(t, x):
    r = x[:3]
    v = x[3:]
    r_mag = np.linalg.norm(r)

    # ODE: r_dot = v, v_dot = -mu/r^3 * r
    dxdt = np.zeros(6)
    dxdt[:3] = v
    dxdt[3:] = -mu / (r_mag**3) * r
    return dxdt

# Initial Conditions (Circular orbit in XY plane)
r0_mag = R_Earth + alt
v0_mag = np.sqrt(mu / r0_mag)

x0 = [r0_mag, 0, 0, 0, v0_mag, 0]

# Time setup (Simulate 3 orbits)
period = 2 * np.pi * np.sqrt(r0_mag**3 / mu)
t_span = (0, 3 * period)
t_eval = np.linspace(0, 3 * period, 1000)

# Solve
sol = solve_ivp(orbital_dynamics, t_span, x0, t_eval=t_eval, rtol=1e-9)

# Plotting
plt.figure(figsize=(6,6))
plt.plot(sol.y[0], sol.y[1], label='ASTERIA Orbit')
plt.xlabel('X (km)')
plt.ylabel('Y (km)')
plt.title('ASTERIA 2-Body Orbital Simulation (ECI)')
plt.axis('equal')
plt.grid(True)
plt.savefig('asteria_orbit_plot.pdf', bbox_inches='tight')
plt.show()
