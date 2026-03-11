import numpy as np

# ASTERIA spec Constants 
ST_NOISE_ARCSEC = 6.0
FGS_NOISE_ARCSEC = 0.1

def arcsec_to_rad(arcsec):
    return (arcsec / 3600.0) * (np.pi / 180.0)

# covariances 
sigma_st = arcsec_to_rad(ST_NOISE_ARCSEC)
R_st = np.eye(3) * (sigma_st**2)

#function to generate noisy measurements 
def get_sensor_reading(true_vec, R):
    # Adding Gaussian noise based on our 3x3 covariance
    noise = np.random.multivariate_normal(np.zeros(3), R)
    noisy_vec = true_vec + noise
    return noisy_vec / np.linalg.norm(noisy_vec)

#Verification Test 
#Running 10000 trials to see if the standard deviation matches 6 arcsec
trials = [get_sensor_reading(np.array([1,0,0]), R_st) for _ in range(10000)]
measured_std = np.std(trials, axis=0)
print(f"Target Sigma (rad): {sigma_st:.2e}")
print(f"Measured Sigma (rad): {measured_std[1]:.2e}") # Should be very close