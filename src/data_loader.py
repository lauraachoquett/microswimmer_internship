import matplotlib.colors as mcolors
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import os
import pyvista as pv
import numpy as np
import time
def sdf_read(path):
    with open(path, "rb") as f:
        L = [float(l) for l in f.readline().decode().split()]
        N = [int(n) for n in f.readline().decode().split()]
        field = np.fromfile(f, dtype=np.float32).reshape(N[::-1])
    h = [l / n for l, n in zip(L, N)]
    return N, h, field


def vel_read(path):
    with open(path, "rb") as f:
        L = [float(l) for l in f.readline().decode().split()]
        N = [int(n) for n in f.readline().decode().split()]
        field = np.fromfile(f, dtype=np.float32).reshape((*N[::-1], 4))
    h = [l / n for l, n in zip(L, N)]
    return N, h, field


import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def load_sdf_from_csv(domain_size):
    path = "/home/lchoquet/project/microswimmer_internship/data/retina.sdf"
    N, h, sdf = sdf_read(path)
    x = np.linspace(0, domain_size[0], N[0])
    y = np.linspace(0, domain_size[1], N[1])
    z = np.linspace(0, domain_size[2], N[2])
    return x, y, z,N, h, sdf


def load_sim_sdf(ratio):
    L = 0.269
    x, y,z, N, _, sdf = load_sdf_from_csv((1, 1, 1))
    
    print("N :",N)
    scale = L / abs(np.min(sdf))
    
    print("SDF :",sdf.shape)
    ## First inerpolation on a standard domain
    sdf_interpolator = RegularGridInterpolator(
        (z,y, x), sdf, bounds_error=False, fill_value=None
    )

    physical_depth = scale * N[2]
    physical_height = scale * N[1]
    physical_width = scale * N[0]
    ## Physical domain used in the simulation : (Here I could change the number of points to refine the grid)
    z_phys = np.linspace(0, physical_depth, N[2])
    y_phys = np.linspace(0, physical_height, N[1])
    x_phys = np.linspace(0, physical_width, N[0])
    
    ## Mesh (Access to x,y,z-coordinates with index)
    X_phys, Y_phys,Z_phys = np.meshgrid(x_phys, y_phys,z_phys, indexing='ij')
    
    ratio_original = N[1] / N[0]
    ratio_physique = physical_height / physical_width

    ## Normalization (values between 0 and 1) to use the previous interpolation function
    Z_norm = Z_phys / physical_depth
    Y_norm = Y_phys / physical_height
    X_norm = X_phys / physical_width
    
    points = np.vstack([Z_norm.ravel(),Y_norm.ravel(), X_norm.ravel()]).T

    ## Compute the sdf on this grid normalized : 
    sdf_interp = sdf_interpolator(points).reshape(X_phys.shape)
    
    ## Rescale the signed distance function 
    sdf_phys = sdf_interp * scale
    
    ## Now we have the interpolate function on the physical domain not the standard one
    sdf_interp_phys = RegularGridInterpolator(
        (x_phys,y_phys,z_phys), sdf_phys, bounds_error=False, fill_value=None
    )

    def sdf_func_phys(point):
        return sdf_interp_phys(point)
    
    ## Same process with the velocity
    path_vel = "data/vel.sdf"
    N, h, vel = vel_read(path_vel)
    v = vel[:, :, :, 0:3]
    vx, vy,vz = v[:, :, :,0], v[:, :, :,1] ,v[:, :,:,2] 
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    
    quantile_999 = np.quantile(speed, 0.999995)
    mask = (speed <= quantile_999)


    # Masquage
    vx = np.where(mask, vx, 0.0)
    vy = np.where(mask, vy, 0.0)
    vz = np.where(mask, vz, 0.0)
    
    vx = np.where(mask, vx, 0.0)
    vy = np.where(mask, vy, 0.0)
    vz = np.where(mask, vz, 0.0)
    
    velocity_interpolator_x = RegularGridInterpolator(
        (z,y, x), vx, bounds_error=False, fill_value=None
    )
    velocity_interpolator_y = RegularGridInterpolator(
        (z,y, x), vy, bounds_error=False, fill_value=None
    )
    velocity_interpolator_z = RegularGridInterpolator(
        (z,y, x), vz, bounds_error=False, fill_value=None
    )
    vx_interp = velocity_interpolator_x(points).reshape(Z_phys.shape)
    vy_interp = velocity_interpolator_y(points).reshape(Z_phys.shape)
    vz_interp = velocity_interpolator_z(points).reshape(Z_phys.shape)
    vx_phys = vx_interp * scale
    vy_phys = vy_interp * scale
    vz_phys = vz_interp * scale

    velocity_interpolator_x_phys = RegularGridInterpolator(
        (x_phys,y_phys,z_phys), vx_phys, bounds_error=False, fill_value=None
    )
    velocity_interpolator_y_phys = RegularGridInterpolator(
        (x_phys,y_phys,z_phys), vy_phys, bounds_error=False, fill_value=None
    )
    velocity_interpolator_z_phys = RegularGridInterpolator(
        (x_phys,y_phys,z_phys), vz_phys, bounds_error=False, fill_value=None
    )

    v_magnitude = np.sqrt(vx_phys**2 + vy_phys**2+ vz_phys**2)
    
    
    def velocity_retina(point):

        return (
            ratio
            * np.array(
                [
                    velocity_interpolator_x_phys(point),
                    velocity_interpolator_y_phys(point),
                    velocity_interpolator_z_phys(point),
                ]
            )
            / np.max(v_magnitude)
        )

    return (
        sdf_func_phys,
        sdf_phys,
        velocity_retina,
        x_phys,
        y_phys,
        z_phys,
        vx_phys,
        vy_phys,
        vz_phys,
        physical_depth,
        physical_width,
        physical_height,
        scale,
    )


def plot_sdf_slices(sdf_phys,X,Y,Z,ix,iy,iz,target_point,start_point,name_fig):


    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(sdf_phys[:, :, iz].T,extent=[X.min(), X.max(), Y.min(), Y.max()],origin='lower', cmap='Reds')
    axes[0].contour(sdf_phys[:,:,iz].T, extent=[X.min(), X.max(), Y.min(), Y.max()],levels=[0], colors='black')
    axes[0].scatter(target_point[0],target_point[1],label='target',color='blue',s=4)
    axes[0].scatter(start_point[0],start_point[1],label='start',color='green',s=4)
    axes[0].set_title(f'Plan Z = {iz} (XY)')
    
    axes[1].imshow(sdf_phys[ix, :, :].T, extent=[Y.min(), Y.max(), Z.min(), Z.max()],origin='lower', cmap='Reds')
    axes[1].contour(sdf_phys[ix,:,:].T,  extent=[Y.min(), Y.max(), Z.min(), Z.max()],levels=[0], colors='black')
    axes[1].scatter(target_point[1],target_point[2],label='target',color='blue',s=4)
    # axes[1].scatter(start_point[1],start_point[2],label='start',color='green',s=4)
    axes[1].set_title(f'Plan X = {ix} (YZ)')
    
    axes[2].imshow(sdf_phys[:, iy, :].T,extent=[X.min(), X.max(), Z.min(), Z.max()], origin='lower', cmap='Reds')
    axes[2].contour(sdf_phys[:,iy,:].T, extent=[X.min(), X.max(), Z.min(), Z.max()],levels=[0], colors='black')
    axes[2].scatter(target_point[0],target_point[2],label='target',color='blue',s=4)
    # axes[2].scatter(start_point[1],start_point[2],label='start',color='green',s=4)
    axes[2].set_title(f'Plan Y = {iy} (XZ)')
    
    
    for ax in axes:
        ax.axis('off')
        
    fig.legend(
        handles=[
            plt.Line2D([0], [0], marker='o', color='w', label='target',
                    markerfacecolor='blue', markersize=5),
            plt.Line2D([0], [0], marker='o', color='w', label='start',
                    markerfacecolor='green', markersize=5)
        ],
        loc='upper center',
        ncol=2,
        frameon=False
    )
    plt.tight_layout()
    save_path_fig = os.path.join('fig',name_fig)
    plt.savefig(save_path_fig,dpi=300,bbox_inches='tight')
    plt.close(fig)
    
    
if __name__ == "__main__":
    # ratio = 5
    # print(z_phys[len(z_phys)//2])
    # start_point = (physical_width * 0.98, physical_height * 0.3,physical_depth*0.5)
    # goal_point = (physical_width*0.3 , physical_height *0.8, physical_depth* 0.5)
    # print(sdf_func_phys(start_point))
    # Nx, Ny, Nz = sdf_phys.shape
    # ix, iy, iz =530, Ny // 4, Nz // 2
    # X, Y,Z = np.meshgrid(x_phys, y_phys,z_phys)
    # plot_sdf_slices(sdf_phys,X,Y,Z,ix,iy,iz,goal_point,start_point,'slices')
    # ratio = 5
    # sdf_func_phys,sdf_phys,velocity_retina,x_phys,y_phys,z_phys,physical_depth,physical_width,physical_height,scale = load_sim_sdf(ratio)
    # grid_size = (len(x_phys), len(y_phys),len(z_phys))
    

    # === Lecture des fichiers ===
    path_vel = "data/vel.sdf"
    N, h, vel = vel_read(path_vel)  # vel.shape = (Nx, Ny, Nz, 4)
    v = vel[:, :, :, :3]            # On garde uniquement vx, vy, vz
    vx, vy, vz = v[:, :, :, 0].T, v[:, :, :, 1].T, v[:, :, :, 2].T
    _, _, sdf = sdf_read('data/retina.sdf')  # sdf.shape doit être (Nx, Ny, Nz)
    sdf=sdf.T
    N=vx.shape
    # === Statistiques initiales ===
    # for name, arr in zip(["vx", "vy", "vz"], [vx, vy, vz]):
    #     print(f"{name}: min={arr.min():.3e}, max={arr.max():.3e}, mean={arr.mean():.3e}, std={arr.std():.3e}")

    speed = np.sqrt(vx**2 + vy**2 + vz**2)

    # === Analyse des vitesses élevées ===
    threshold = 100  # à adapter selon le cas
    outliers = np.argwhere(speed > threshold)

    # === Histogramme de la vitesse ===
    os.makedirs("fig", exist_ok=True)
    plt.hist(speed.flatten(), bins=100)
    plt.yscale("log")
    plt.title("Histogramme des vitesses")
    plt.xlabel("||v||")
    plt.ylabel("Fréquence (log)")
    plt.savefig('fig/hist_v')
    plt.close()
    
    quantile_999 = np.quantile(speed, 0.999995)
    # Créer un masque pour ne garder que les vitesses < seuil
    mask = (speed <= quantile_999)


    # Masquage
    vx = np.where(mask, vx, 0.0)
    vy = np.where(mask, vy, 0.0)
    vz = np.where(mask, vz, 0.0)

    
    filtered_speed = np.sqrt(vx**2 + vy**2 + vz**2)
    print(f"After filtering: max={filtered_speed.max():.3e}, min={filtered_speed.min():.3e}, mean={filtered_speed.mean():.3e}, std={filtered_speed.std():.3e}")
    for name, arr in zip(["vx", "vy", "vz","sdf"], [vx, vy, vz,sdf]):
        print(f"After filtering {name}:  max={arr.max():.3e}, min={arr.min():.3e}, mean={arr.mean():.3e}, std={arr.std():.3e}")
    nx,ny,nz = N
    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)

    # Ajout des champs
    print("shape:")
    print(N)
    print(vx.shape)
    print(vy.shape)
    print(vz.shape)
    print(sdf.shape)
    print(nx*ny*nz)
    grid["SDF"] = sdf.flatten(order="F")
    print(grid["SDF"].shape)
    
    velocity_vectors = np.stack([vx, vy, vz], axis=-1)
    grid["velocity"] = velocity_vectors.reshape(-1, 3, order="F")
    print(grid["velocity"].shape)  # doit être (nx * ny * nz, 3)
    # Sauvegarde
    output_path = f"velocity_with_sdf_{time.time()}.vti"
    grid.save(output_path)
    print(f"Fichier sauvegardé dans : {output_path}")
    
    
    
    
    
    # # Création de la grille uniforme physique
    # dx = x_phys[1] - x_phys[0]
    # dy = y_phys[1] - y_phys[0]
    # dz = z_phys[1] - z_phys[0]
    # grid = pv.ImageData()
    # grid.dimensions = (nx, ny, nz)
    # grid.spacing = (dx, dy, dz)
    # grid.origin = (0, 0, 0)

    # # Ajout des champs
    # grid["SDF"] = sdf_phys.flatten(order="F")
    # velocity_vectors = np.stack([vx, vy, vz], axis=-1)
    # print(velocity_vectors.shape)
    # grid["velocity"] = velocity_vectors.reshape(-1, 3, order="F")
    # print(grid["velocity"].shape)  # doit être (nx * ny * nz, 3)
    # # Sauvegarde
    # output_path = "velocity_with_sdf.vti"
    # grid.save(output_path)
    # print(f"Fichier sauvegardé dans : {output_path}")
        
    