import matplotlib.colors as mcolors
import numpy as np
from scipy.interpolate import RegularGridInterpolator


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


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

def load_sdf_from_csv(domain_size):
    path = "/home/lchoquet/project/microswimmer_internship/data/retina.sdf"
    N, h, sdf = sdf_read(path)
    sdf = sdf[N[2] // 2, :, :]
    x = np.linspace(0, domain_size[0], N[0])
    y = np.linspace(0, domain_size[1], N[1])
    return x, y, N, h, sdf

def load_sim_sdf(ratio):
    L = 0.269
    x, y, N, _, sdf = load_sdf_from_csv((1, 1))
    scale = L / abs(np.min(sdf))
    sdf_interpolator = RegularGridInterpolator(
        (y, x), sdf, bounds_error=False, fill_value=None
    )
    

    physical_height = scale * N[1]
    physical_width = scale  * N[0]
    
    y_phys = np.linspace(0, physical_height, N[1])
    x_phys = np.linspace(0, physical_width, N[0])
    X_phys, Y_phys = np.meshgrid(x_phys, y_phys)
    
    
    ratio_original = N[1] / N[0]
    ratio_physique =   physical_height / physical_width
    
    Y_norm = Y_phys / physical_height  
    X_norm = X_phys / physical_width   
    
    points = np.vstack([Y_norm.ravel(), X_norm.ravel()]).T
    
    sdf_interp = sdf_interpolator(points).reshape(Y_phys.shape)
    
    sdf_phys = sdf_interp * scale
    sdf_interp_phys = RegularGridInterpolator(
        (y_phys, x_phys), sdf_phys, bounds_error=False, fill_value=None
    )
    def sdf_func_phys(point):
        return(sdf_interp_phys(point[::-1]))
    
    path_vel = "data/vel.sdf"
    N, h, vel = vel_read(path_vel)
    v = vel[N[2] // 2, :, :, 0:2]
    vx, vy = v[:, :, 0], v[:, :, 1]
    velocity_interpolator_x = RegularGridInterpolator(
        (y, x), vx, bounds_error=False, fill_value=None
    )
    velocity_interpolator_y = RegularGridInterpolator(
        (y, x), vy, bounds_error=False, fill_value=None
    )
    vx_interp =  velocity_interpolator_x(points).reshape(Y_phys.shape)
    vy_interp =  velocity_interpolator_y(points).reshape(Y_phys.shape)
    vx_phys = vx_interp * scale
    vy_phys = vy_interp * scale

    velocity_interpolator_x_phys= RegularGridInterpolator(
        (y_phys, x_phys), vx_phys, bounds_error=False, fill_value=None
    )
    velocity_interpolator_y_phys = RegularGridInterpolator(
        (y_phys, x_phys), vy_phys, bounds_error=False, fill_value=None
    )

    v_magnitude = np.sqrt(vx_phys**2 + vy_phys**2)
    def velocity_retina(point):
        
        return (
            ratio
            * np.array(
                [
                    velocity_interpolator_x_phys(point[::-1]),
                    velocity_interpolator_y_phys(point[::-1]),
                ]
            )
            / np.max(v_magnitude)
        )
    return sdf_func_phys,velocity_retina,x_phys,y_phys, physical_width,physical_height,scale















if __name__ == "__main__":
    L = 0.269
    
    x, y, N, _, sdf = load_sdf_from_csv((1, 1))
    
    scale = L / abs(np.min(sdf))
    print("Scale :", scale)
    print("N :", N)
    print("Min SDF value:", np.min(sdf))
    print("SDF shape after slice:", sdf.shape)
    
    sdf_interpolator = RegularGridInterpolator(
        (y, x), sdf, bounds_error=False, fill_value=None
    )

 

    physical_height = scale * N[1]
    physical_width = scale  * N[0]
    print('physical height :',physical_height)
    print('physical width :',physical_width)
    
    y_phys = np.linspace(0, physical_height, N[1])
    x_phys = np.linspace(0, physical_width, N[0])
    X_phys, Y_phys = np.meshgrid(x_phys, y_phys)
    
    
    ratio_original = N[1] / N[0]
    ratio_physique =   physical_height / physical_width
    print(f"Ratio d'aspect original (points) : {ratio_original}")
    print(f"Ratio d'aspect physique (unités) : {ratio_physique}")
    
    
    Y_norm = Y_phys / physical_height  
    X_norm = X_phys / physical_width   
    
    
    points = np.vstack([Y_norm.ravel(), X_norm.ravel()]).T
    
    
    sdf_interp = sdf_interpolator(points).reshape(Y_phys.shape)
    
    
    sdf_phys = sdf_interp * scale
    
    sdf_interp_phys = RegularGridInterpolator(
        (y_phys, x_phys), sdf_phys, bounds_error=False, fill_value=None
    )
    def sdf_func_phys(point):
        return(sdf_interp_phys(point[::-1]))
    
    print("Shape new sdf:", sdf_phys.shape)
    print("Min new sdf:", np.min(sdf_phys))
    
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X_phys, Y_phys, sdf_phys, levels=100, cmap="Reds")
    cbar = plt.colorbar(contour)
    cbar.set_label("Signed Distance (Physical Units)", fontsize=12)
    plt.contour(X_phys, Y_phys, sdf_phys, levels=[0], colors="black", linewidths=1)
    
    start_point = (physical_width * 0.98, physical_height * 0.3)
    goal_point = (physical_width * 0.54, physical_height * 0.675)
    
    print(sdf_func_phys(start_point))
    print(sdf_func_phys(goal_point))
    
    
    plt.scatter(goal_point[0], goal_point[1], color="black", s=5)
    plt.scatter(start_point[0], start_point[1], label="Start", color="blue", s=5)
    plt.axis("equal")  
    plt.legend()
    plt.savefig('fig/test_scale_corrected.png', dpi=300, bbox_inches='tight')
    
    print(f"Valeur SDF min physique: {np.min(sdf_phys)}, Target: {-L}")
    print(f"dx : {x_phys[1]-x_phys[0]} and dy : {y_phys[1]-y_phys[0]}")
    # np.save("/home/lchoquet/project/microswimmer_internship/data/vel.sdf", sdf_new, allow_pickle=False)

    path_vel = "data/vel.sdf"
    N, h, vel = vel_read(path_vel)
    v = vel[N[2] // 2, :, :, 0:2]
    vx, vy = v[:, :, 0], v[:, :, 1]
    velocity_interpolator_x = RegularGridInterpolator(
        (y, x), vx, bounds_error=False, fill_value=None
    )
    velocity_interpolator_y = RegularGridInterpolator(
        (y, x), vy, bounds_error=False, fill_value=None
    )
    
    vx_interp =  velocity_interpolator_x(points).reshape(Y_phys.shape)
    vy_interp =  velocity_interpolator_y(points).reshape(Y_phys.shape)
    vx_phys = vx_interp * scale
    vy_phys = vy_interp * scale
    v_magnitude = np.sqrt(vx_phys**2 + vy_phys**2)
    print("norm V_max : ", np.max(v_magnitude))
    velocity_interpolator_x_phys= RegularGridInterpolator(
        (y_phys, x_phys), vx_phys, bounds_error=False, fill_value=None
    )
    velocity_interpolator_y_phys = RegularGridInterpolator(
        (y_phys, x_phys), vy_phys, bounds_error=False, fill_value=None
    )

    def velocity_retina(point):
        
        return (
            ratio
            * np.array(
                [
                    velocity_interpolator_x_phys(point[::-1]),
                    velocity_interpolator_y_phys(point[::-1]),
                ]
            )
            / np.max(v_magnitude))
    fig, axes = plt.subplots(ncols=2, figsize=(14, 6))

    # Add contours for the signed distance function (SDF) on both subplots
    axes[0].contour(X_phys, Y_phys, sdf_phys, levels=[0], colors="black", linewidths=1, zorder=10)
    axes[1].contour(X_phys, Y_phys, sdf_phys, levels=[0], colors="black", linewidths=1, zorder=10)

    # Plot velocity components as filled contours
    vxmin = np.min(vx_phys)
    vxmax = np.max(vx_phys)
    normx = mcolors.TwoSlopeNorm(vmin=vxmin, vcenter=0, vmax=vxmax)
    vymin = np.min(vy_phys)
    vymax = np.max(vy_phys)
    normy = mcolors.TwoSlopeNorm(vmin=vymin, vcenter=0, vmax=vymax)

    contourx = axes[0].contourf(X_phys, Y_phys, vx_phys, levels=100, cmap="RdBu",norm=normx)
    contoury = axes[1].contourf(X_phys, Y_phys, vy_phys, levels=100, cmap="RdBu",norm=normy)

    # Add colorbars for the velocity components
    cbarx = fig.colorbar(contourx, ax=axes[0], orientation="vertical")
    cbary = fig.colorbar(contoury, ax=axes[1], orientation="vertical")
    axes[0].set_title("Velocity X", fontsize=14)
    axes[1].set_title("Velocity Y", fontsize=14)

    # Remove axes
    axes[0].axis("off")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("fig/retina_example_velocity.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
