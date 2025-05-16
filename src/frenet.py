import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Courbe : hélice
# t = np.linspace(0, 4 * np.pi, 200)
# x = np.cos(t)
# y = np.sin(t)
# z = t / 2
# curve = np.vstack((x, y, z)).T
# print('curve : ',curve.shape)
# Calcul des vecteurs de Frenet : T, N, B
def compute_frenet_frame_bis(path, dim):
    dt = np.gradient(path, axis=0)
    dt_norm = np.linalg.norm(dt, axis=1, keepdims=True)
    T = dt / dt_norm

    if dim == 2:
        return T, None, None
    elif dim == 3:
        dT = np.gradient(T, axis=0)
        dT_norm = np.linalg.norm(dT, axis=1, keepdims=True)

        # Pour éviter la division par 0 : on remplace les trop petites normes par 1
        threshold_norm = 1e-6
        dT_norm[dT_norm < threshold_norm] = 1.0  # pour éviter de diviser par 0
        N = dT / dT_norm

        # Optionnel : pour les endroits où ||dT|| < seuil, on peut aussi mettre N = vecteur nul
        N[dT_norm < threshold_norm] = 0.0

        B = np.cross(T, N)

        # On peut aussi normaliser B
        B_norm = np.linalg.norm(B, axis=1, keepdims=True)
        B[B_norm > 0] /= B_norm[B_norm > 0]  # évite les divisions par 0
        return T, N, B

def compute_frenet_frame(path, dim):
    dt = np.gradient(path, axis=0)
    dt_norm = np.linalg.norm(dt, axis=1, keepdims=True)
    T = dt / (dt_norm + 1e-8)  # évite division par zéro

    if dim == 2:
        return T, None, None

    elif dim == 3:
        N = np.zeros_like(path)
        B = np.zeros_like(path)

        # Initialisation : on prend un vecteur arbitraire non colinéaire à T[0]
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
                B[i] = B[i - 1]

        return T, N, B
    
# dim=3
# T, N, B = compute_frenet_frame_bis(curve,dim)

# print("Start video")


# # Animation
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x, y, z, 'gray', linewidth=1, label="Courbe")

# ax.set_xlim([-1.5, 1.5])
# ax.set_ylim([-1.5, 1.5])
# ax.set_zlim([0, max(z)])
# ax.set_title("Repère de Frenet le long d'une hélice")
# ax.legend()

# # Stocker les quivers pour pouvoir les supprimer à chaque frame
# quivers = []

# def update(i):
#     global quivers
#     # Supprimer les anciens quivers
#     for q in quivers:
#         q.remove()
#     quivers = []

#     p = curve[i]
#     t_vec = T[i]
#     n_vec = N[i]
#     b_vec = B[i]

#     quivers.append(ax.quiver(*p, *t_vec, color='r'))
#     quivers.append(ax.quiver(*p, *n_vec, color='g'))
#     quivers.append(ax.quiver(*p, *b_vec, color='b'))

#     return quivers

# ani = FuncAnimation(fig, update, frames=len(curve), interval=50)

# # Sauvegarde en vidéo
# writer = FFMpegWriter(fps=30)
# ani.save("fig/frenet.mp4", writer=writer, dpi=300)