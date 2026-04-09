import numpy as np
from scipy.linalg import svd

# stare tracker noise
sigma_cross = (6.0 / 3600.0) * (np.pi / 180.0) # 6 arcsec
sigma_bore = (40.0 / 3600.0) * (np.pi / 180.0) # 40 arcsec
R_cov = np.diag([sigma_cross**2, sigma_cross**2, sigma_bore**2])

def solve_triad(r1, r2, b1, b2):
    v1, v2 = b1, np.cross(b1, b2)/np.linalg.norm(np.cross(b1, b2))
    v3 = np.cross(v1, v2)
    w1, w2 = r1, np.cross(r1, r2)/np.linalg.norm(np.cross(r1, r2))
    w3 = np.cross(w1, w2)
    return np.column_stack((v1, v2, v3)) @ np.column_stack((w1, w2, w3)).T

def solve_svd(r_vectors, b_vectors):
    B = sum(np.outer(b, r) for b, r in zip(b_vectors, r_vectors))
    U, S, Vt = svd(B)
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    return R

# monte carlo sim
trials = 1000
errors_svd, errors_triad = [], []
np.random.seed(42)

for _ in range(trials):
    R_true = np.eye(3)
    r1, r2 = np.array([1,0,0]), np.array([0,1,0])

    b1 = r1 + np.random.multivariate_normal(np.zeros(3), R_cov)
    b2 = r2 + np.random.multivariate_normal(np.zeros(3), R_cov)
    b1 /= np.linalg.norm(b1)
    b2 /= np.linalg.norm(b2)

    R_svd = solve_svd([r1, r2], [b1, b2])
    R_triad = solve_triad(r1, r2, b1, b2)

    errors_svd.append(np.degrees(np.arccos(np.clip((np.trace(R_svd @ R_true.T) - 1)/2, -1, 1))))
    errors_triad.append(np.degrees(np.arccos(np.clip((np.trace(R_triad @ R_true.T) - 1)/2, -1, 1))))

print(f"SVD Mean Error: {np.mean(errors_svd):.6f} deg")
print(f"TRIAD Mean Error: {np.mean(errors_triad):.6f} deg")
