import numpy as np
from scipy.linalg import expm

#Grabbing our perfect HW1 inertia matrix 
J_nominal = np.diag([0.0867, 0.1817, 0.1117])
D_vals, V = np.linalg.eigh(J_nominal)

#Perturb the eigenvalues by a few percent 
d = np.random.normal(0, 0.02, 3)
D_tilde = np.diag(D_vals * (1 + d))

#Twist the axes by a random rotation vector 
v_vec = np.random.normal(0, np.radians(2), 3) 
v_hat = np.array([[0, -v_vec[2], v_vec[1]], 
                  [v_vec[2], 0, -v_vec[0]], 
                  [-v_vec[1], v_vec[0], 0]])
V_tilde = V @ expm(v_hat)

#The final "Messy" Inertia Matrix
J_perturbed = V_tilde @ D_tilde @ V_tilde.T
print(J_perturbed)

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Using the J_perturbed we just generated above
h_w_val = 1.047 * (1.2 * J_perturbed[2,2] - J_perturbed[0,0])
h_w = np.array([h_w_val, 0, 0]) # Rotor along b1 (minor axis)

# the rotor is a 

def gyrostat_ode(t, omega):
    H_sys = J_perturbed @ omega + h_w
    omega_dot = np.linalg.inv(J_perturbed) @ (-np.cross(omega, H_sys))
    return omega_dot

# Initial conditions: 10 RPM spin + slight wobble
omega_0 = [1.047, 0.01, 0.01]
t_eval = np.linspace(0, 50, 1000)
sol = solve_ivp(gyrostat_ode, (0, 50), omega_0, t_eval=t_eval)

# The Stability Diagram
plt.figure(figsize=(6, 6))
# Plotting w2 vs w3 (the axes we AREN'T spinning around)
plt.plot(sol.y[1], sol.y[2], color='purple', label='Nutation Path')
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.title('Transverse Phase Portrait (Stability Check)')
plt.xlabel('$\omega_2$ (rad/s)')
plt.ylabel('$\omega_3$ (rad/s)')
plt.axis('equal') # This makes the circle look like a circle
plt.grid(True, linestyle='--')
plt.legend()
plt.savefig('Stability_Diagram.png', dpi=300)
plt.show()