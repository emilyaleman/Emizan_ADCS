import numpy as np
import matplotlib.pyplot as plt

ST_NOISE_ARCSEC = 6.0      # Star tracker cross-axis accuracy
FGS_NOISE_ARCSEC = 0.1     # Fine guidance sensor accuracy

N_TRIALS = 10000

def arcsec_to_rad(arcsec):
    """
    Convert arcseconds to radians
    """
    return (arcsec / 3600.0) * (np.pi / 180.0)


sigma_st = arcsec_to_rad(ST_NOISE_ARCSEC)
sigma_fgs = arcsec_to_rad(FGS_NOISE_ARCSEC)

print("Sensor Noise (Radians)")
print("----------------------")
print("Star Tracker sigma:", sigma_st)
print("FGS sigma:", sigma_fgs)
print()



# Covariance Matrices
R_st = np.eye(3) * sigma_st**2
R_fgs = np.eye(3) * sigma_fgs**2



# 4. Sensor Measurement Function
def get_sensor_reading(true_vec, sigma_rad):
    """
    Generate a noisy unit vector by applying a small random rotation.
    sigma_rad: 1-sigma angular noise in radians
    """
    # Draw two small rotation angles about axes perpendicular to true_vec
    # (cross-axis errors)
    angle1 = np.random.normal(0, sigma_rad)
    angle2 = np.random.normal(0, sigma_rad)

    # Build two axes perpendicular to true_vec
    arbitrary = np.array([1, 0, 0]) if abs(true_vec[1]) > 0.1 or abs(true_vec[2]) > 0.1 \
                else np.array([0, 1, 0])
    perp1 = np.cross(true_vec, arbitrary)
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(true_vec, perp1)
    perp2 /= np.linalg.norm(perp2)

    # Apply small-angle rotations using Rodrigues' formula
    def rotate(v, axis, angle):
        return (v * np.cos(angle) +
                np.cross(axis, v) * np.sin(angle) +
                axis * np.dot(axis, v) * (1 - np.cos(angle)))

    noisy_vec = rotate(true_vec, perp1, angle1)
    noisy_vec = rotate(noisy_vec, perp2, angle2)
    return noisy_vec
    
# 5. Monte Carlo Simulation
true_vec = np.array([1.0, 0.0, 0.0])

st_measurements = []
fgs_measurements = []

for _ in range(N_TRIALS):

    st_measurements.append(get_sensor_reading(true_vec, R_st))
    fgs_measurements.append(get_sensor_reading(true_vec, R_fgs))


st_measurements = np.array(st_measurements)
fgs_measurements = np.array(fgs_measurements)



# 6. Compute Statistics
st_std = np.std(st_measurements, axis=0)
fgs_std = np.std(fgs_measurements, axis=0)

print("Monte Carlo Results")
print("----------------------")

print("Star Tracker")
print("Target sigma:", sigma_st)
print("Measured sigma:", st_std)

print()

print("FGS")
print("Target sigma:", sigma_fgs)
print("Measured sigma:", fgs_std)

# 7. Optional Visualization


errors = st_measurements - true_vec

plt.figure(figsize=(6,4))
plt.hist(errors[:,1], bins=60)
plt.title("Star Tracker Measurement Error (Y axis)")
plt.xlabel("Error")
plt.ylabel("Count")

plt.show()
