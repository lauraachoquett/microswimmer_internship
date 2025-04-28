import numpy as np
from scipy.interpolate import RegularGridInterpolator


def sdf_read(path):
    with open(path, "rb") as f:
        L = [float(l) for l in f.readline().decode().split()]
        N = [int(n) for n in f.readline().decode().split()]
        field = -np.fromfile(f, dtype=np.float32).reshape(N[::-1])
    h = [l / n for l, n in zip(L, N)]
    return N, h, field


def vel_read(path):
    with open(path, "rb") as f:
        L = [float(l) for l in f.readline().decode().split()]
        N = [int(n) for n in f.readline().decode().split()]
        field = np.fromfile(f, dtype=np.float32).reshape((*N[::-1], 4))
    h = [l / n for l, n in zip(L, N)]
    return N, h, field


def load_sdf_from_csv(domain_size):
    path = 'data/retina.sdf'
    N, h, sdf = sdf_read(path)
    sdf = sdf[N[2]//2,:,:]
    x = np.linspace(0,domain_size[0],N[0])
    y = np.linspace(0,domain_size[1],N[1])
    return x,y,N,h,sdf
    
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    domain_size =(1,1)

    x, y, N, _, sdf = load_sdf_from_csv(domain_size)

    print('SDF shape after slice :', sdf.shape)
    fig, ax = plt.subplots(figsize=(15, 7))
    grid_size = (N[0], N[1])

    start_point = (0.98, 0.3)
    goal_point = (0.44, 0.55)
    sdf_interpolator = RegularGridInterpolator((y, x), sdf, bounds_error=False, fill_value=None)

    def sdf_function(point):
        return sdf_interpolator(point[::-1])  # Inverser car l'interpolateur attend (y, x)

    print(sdf_function(start_point))
    print(sdf_function(goal_point))

    X, Y = np.meshgrid(x, y)
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, sdf, levels=100, cmap="viridis")
    cbar = plt.colorbar(contour)
    cbar.set_label("Signed Distance", fontsize=12)
    plt.contour(X, Y, sdf, levels=[0], colors="black", linewidths=1)
    plt.scatter(goal_point[0], goal_point[1], label='Goal', color='red', s=5)
    plt.scatter(start_point[0], start_point[1], label='Start', color='blue', s=5)
    plt.axis("off")
    plt.legend(fontsize=10)
    plt.savefig('fig/retina_example.png', dpi=200, bbox_inches='tight')
    plt.close()
    np.save("data/retina2D.npy", sdf, allow_pickle=False)

    path_vel = 'data/vel.sdf'
    n, h, vel = vel_read(path_vel)
    print(vel.shape)
    v = vel[n[2]//2,:,:,0:2]
    vx = np.flip(v[:,:,0],axis=0)
    vy = np.flip(v[:,:,1],axis=0)
    v = np.sqrt(vx**2 + vy**2)
    print("max :" ,np.max(v))
    velocity_interpolator_x = RegularGridInterpolator((y, x), vx, bounds_error=False, fill_value=None)
    velocity_interpolator_y = RegularGridInterpolator((y, x), vy, bounds_error=False, fill_value=None)
    def velocity_retina(point):
        return np.array([velocity_interpolator_x(point[::-1]),velocity_interpolator_y(point[::-1])])
    
    np.save("vel2D.npy", v, allow_pickle=False)
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
    im0 = axes[0].imshow(vx, cmap="viridis")
    im1 = axes[1].imshow(vy, cmap="viridis")
    cbar0 = fig.colorbar(im0, ax=axes[0], orientation="vertical")
    cbar0.set_label("Velocity X", fontsize=12)
    cbar1 = fig.colorbar(im1, ax=axes[1], orientation="vertical")
    cbar1.set_label("Velocity Y", fontsize=12)
    axes[0].set_title("Velocity X", fontsize=14)
    axes[1].set_title("Velocity Y", fontsize=14)
    plt.tight_layout()
    plt.savefig('fig/retina_example_velocity.png',dpi=200,bbox_inches='tight')