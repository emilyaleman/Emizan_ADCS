import numpy as np

# 1. Mass and Dimensions (kg and meters)
m_bus = 10.0
m_panels = 1.0  # 0.5 kg each
total_mass = m_bus + m_panels

# Dimensions aligned to your axes: [b1, b2, b3]
# b1=20cm, b2=10cm, b3=30cm
bus_d = np.array([0.2, 0.1, 0.3]) 
# Panel dimensions: extends along b1, height b2, length b3
panel_d = np.array([0.3, 0.005, 0.2]) 

# 2. Local Inertia (Box about its own CoM)
def get_i_local(m, d):
    return (m / 12) * np.array([
        [d[1]**2 + d[2]**2, 0, 0],
        [0, d[0]**2 + d[2]**2, 0],
        [0, 0, d[0]**2 + d[1]**2]
    ])

I_bus = get_i_local(m_bus, bus_d)

# 3. Parallel Axis Theorem for Panels
# Panels are at +/- (bus_width/2 + panel_width/2) along b1
r_val = (0.2/2) + (0.3/2) 
r_p1 = np.array([r_val, 0, 0])
r_p2 = np.array([-r_val, 0, 0])

def parallel_axis(I_loc, m, r):
    return I_loc + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))

I_p1 = parallel_axis(get_i_local(0.5, panel_d), 0.5, r_p1)
I_p2 = parallel_axis(get_i_local(0.5, panel_d), 0.5, r_p2)

J_total = I_bus + I_p1 + I_p2

print("Updated Inertia Matrix (J):")
print(np.round(J_total, 4))