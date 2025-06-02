import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


def compute_frenet_frame(path, dim):
    dt = np.gradient(path, axis=0)
    dt_norm = np.linalg.norm(dt, axis=1, keepdims=True)
    T = dt / (dt_norm + 1e-8)  

    if dim == 2:
        return T, None, None

    elif dim == 3:
        N = np.zeros_like(path)
        B = np.zeros_like(path)

        T0 = T[0]
        ref = np.array([1.0, 0.0, 0.0])
        if np.allclose(T0, ref):
            ref = np.array([0.0, 1.0, 0.0])
        N0 = np.cross(T0, ref)
        N0 /= np.linalg.norm(N0)
        B0 = np.cross(T0, N0)

        N[0] = N0
        B[0] = B0

        for i in range(1, len(path)):
            dT = T[i] - T[i - 1]
            norm_dT = np.linalg.norm(dT)

            if norm_dT > 1e-6:
                N[i] = dT / norm_dT
                B[i] = np.cross(T[i], N[i])
                norm_B = np.linalg.norm(B[i])
                if norm_B > 1e-6:
                    B[i] /= norm_B
                    N[i] = np.cross(B[i], T[i])  # Re-orthonormalise
            else:
                N[i] = N[i - 1]                
                B[i] = np.cross(T[i], N[i])        
                B[i] /= (np.linalg.norm(B[i]) + 1e-8)  
                N[i] = np.cross(B[i], T[i])

        return T, N, B


import numpy as np

def normalize(v):
    return v / np.linalg.norm(v)

def double_reflection_rmf(points):
    points = np.array(points)
    n = len(points)

    # Tangents
    tangents = [normalize(points[i+1] - points[i]) for i in range(n - 1)]
    
    # Initial frame: choose arbitrary normal N0 orthogonal to T0
    T0 = tangents[0]
    arbitrary = np.array([0, 0, 1])
    if np.allclose(np.cross(T0, arbitrary), 0):
        arbitrary = np.array([1, 0, 0])
    N0 = normalize(np.cross(np.cross(T0, arbitrary), T0))
    B0 = np.cross(T0, N0)

    frames = [(T0, N0, B0)]
    N_prev = N0

    for i in range(1, n - 1):
        v1 = tangents[i - 1]
        v2 = tangents[i]

        # First reflection
        r = v2 + v1
        if np.linalg.norm(r) < 1e-10:  # 180° turn: reset
            N_curr = N_prev
        else:
            r = normalize(r)
            N_prime = N_prev - 2 * np.dot(N_prev, r) * r

            # Second reflection
            r = v2
            N_curr = N_prime - 2 * np.dot(N_prime, r) * r

        B_curr = np.cross(v2, N_curr)
        frames.append((v2, N_curr, B_curr))
        N_prev = N_curr
    frames.append((v2, N_curr, B_curr))
    # Convert frames to arrays
    T = np.array([f[0] for f in frames])
    N = np.array([f[1] for f in frames])
    B = np.array([f[2] for f in frames])
    return T, N, B