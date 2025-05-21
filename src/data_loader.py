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
    sdf_interpolator = RegularGridInterpolator(
        (z,y, x), sdf, bounds_error=False, fill_value=None
    )

    physical_depth = scale * N[2]
    physical_height = scale * N[1]
    physical_width = scale * N[0]

    z_phys = np.linspace(0, physical_depth, N[2])
    y_phys = np.linspace(0, physical_height, N[1])
    x_phys = np.linspace(0, physical_width, N[0])
    X_phys, Y_phys,Z_phys = np.meshgrid(x_phys, y_phys,z_phys, indexing='ij')
    
    ratio_original = N[1] / N[0]
    ratio_physique = physical_height / physical_width

    Z_norm = Z_phys / physical_depth
    Y_norm = Y_phys / physical_height
    X_norm = X_phys / physical_width
    points = np.vstack([Z_norm.ravel(),Y_norm.ravel(), X_norm.ravel()]).T

    sdf_interp = sdf_interpolator(points).reshape(X_phys.shape)
    sdf_phys = sdf_interp * scale
    sdf_interp_phys = RegularGridInterpolator(
        (x_phys,y_phys,z_phys), sdf_phys, bounds_error=False, fill_value=None
    )

    def sdf_func_phys(point):
        return sdf_interp_phys(point)
    path_vel = "data/vel.sdf"
    N, h, vel = vel_read(path_vel)
    v = vel[:, :, :, 0:3]
    vx, vy,vz = v[:, :, :,0], v[:, :, :,1] ,v[:, :,:,2] 

    velocity_interpolator_x = RegularGridInterpolator(
        (z,y, x), vx, bounds_error=False, fill_value=None
    )
    velocity_interpolator_y = RegularGridInterpolator(
        (z,y, x), vy, bounds_error=False, fill_value=None
    )
    velocity_interpolator_z = RegularGridInterpolator(
        (z,y, x), vy, bounds_error=False, fill_value=None
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
        velocity_retina,
        x_phys,
        y_phys,
        z_phys,
        physical_depth,
        physical_width,
        physical_height,
        scale,
    )


if __name__ == "__main__":
    ratio = 5
    sdf_func_phys,velocity_retina,x_phys,y_phys,z_phys,physical_depth,physical_width,physical_height,scale = load_sim_sdf(ratio)
    print(z_phys[len(z_phys)//2])
    start_point = (physical_width * 0.98, physical_height * 0.3,physical_depth*0.5)
    start_point = (physical_width * 0.98, physical_height * 0.3,physical_depth*0.5)
    print(sdf_func_phys(start_point))
    # fig, axes = plt.subplots(ncols=2, figsize=(14, 6))

    # # Add contours for the signed distance function (SDF) on both subplots
    # axes[0].contour(
    #     X_phys, Y_phys, sdf_phys, levels=[0], colors="black", linewidths=1, zorder=10
    # )
    # axes[1].contour(
    #     X_phys, Y_phys, sdf_phys, levels=[0], colors="black", linewidths=1, zorder=10
    # )

    # # Plot velocity components as filled contours
    # vxmin = np.min(vx_phys)
    # vxmax = np.max(vx_phys)
    # normx = mcolors.TwoSlopeNorm(vmin=vxmin, vcenter=0, vmax=vxmax)
    # vymin = np.min(vy_phys)
    # vymax = np.max(vy_phys)
    # normy = mcolors.TwoSlopeNorm(vmin=vymin, vcenter=0, vmax=vymax)

    # contourx = axes[0].contourf(
    #     X_phys, Y_phys, vx_phys, levels=100, cmap="RdBu", norm=normx
    # )
    # contoury = axes[1].contourf(
    #     X_phys, Y_phys, vy_phys, levels=100, cmap="RdBu", norm=normy
    # )

    # # Add colorbars for the velocity components
    # cbarx = fig.colorbar(contourx, ax=axes[0], orientation="vertical")
    # cbary = fig.colorbar(contoury, ax=axes[1], orientation="vertical")
    # axes[0].set_title("Velocity X", fontsize=14)
    # axes[1].set_title("Velocity Y", fontsize=14)

    # # Remove axes
    # axes[0].axis("off")
    # axes[1].axis("off")
    # plt.tight_layout()
    # plt.savefig("fig/retina_example_velocity.png", dpi=400, bbox_inches="tight")
    # plt.close(fig)
