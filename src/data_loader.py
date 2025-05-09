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


def load_sdf_from_csv(domain_size):
    path = "/home/lchoquet/project/microswimmer_internship/data/retina.sdf"
    N, h, sdf = sdf_read(path)
    sdf = sdf[N[2] // 2, :, :]
    x = np.linspace(0, domain_size[0], N[0])
    y = np.linspace(0, domain_size[1], N[1])
    return x, y, N, h, sdf


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    domain_size = (1, 1)

    x, y, N, _, sdf = load_sdf_from_csv(domain_size)
    print(np.min(sdf))
    print("SDF shape after slice :", sdf.shape)
    fig, ax = plt.subplots(figsize=(15, 7))
    grid_size = (N[0], N[1])

    start_point = (0.98, 0.3)

    sdf_interpolator = RegularGridInterpolator(
        (y, x), sdf, bounds_error=False, fill_value=None
    )

    def sdf_function(point):
        return sdf_interpolator(
            point[::-1]
        )  # Inverser car l'interpolateur attend (y, x)

    point_list = np.random.rand(20, 2)

    goal_points = [
        point
        for point in point_list
        if (sdf_function(point) < -0.15 and point[0] < 0.95)
    ]
    goal_points = [(0.54, 0.675)]

    x_new = np.linspace(0, domain_size[0], N[0] * 5)
    y_new = np.linspace(0, domain_size[1], N[1] * 5)
    X_new, Y_new = np.meshgrid(x_new, y_new)
    X, Y = np.meshgrid(x, y)
    print("Acc_x : ", 1 / (N[0] * 5))
    print("Acc_y : ", 1 / (N[1] * 5))
    sdf_new = sdf_interpolator((Y_new, X_new))
    print(sdf_new.shape)
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, sdf, levels=100, cmap="Reds")
    cbar = plt.colorbar(contour)
    cbar.set_label("Signed Distance", fontsize=12)
    plt.contour(X, Y, sdf, levels=[0], colors="black", linewidths=1)
    for goal_point in goal_points:
        plt.scatter(goal_point[0], goal_point[1], color="black", s=5)
    plt.scatter(start_point[0], start_point[1], label="Start", color="blue", s=5)
    plt.axis("off")
    plt.legend()
    plt.savefig("../fig/retina_example_test.png", dpi=400, bbox_inches="tight")
    plt.close()
    # np.save("/home/lchoquet/project/microswimmer_internship/data/vel.sdf", sdf_new, allow_pickle=False)

    # path_vel = "/home/lchoquet/project/microswimmer_internship/data/vel.sdf"
    # n, h, vel = vel_read(path_vel)
    # print(vel.shape)
    # v = vel[n[2] // 2, :, :, 0:2]
    # vx = v[:, :, 0]
    # vy = v[:, :, 1]
    # v = np.sqrt(vx**2 + vy**2)
    # print("max :", np.max(v))
    # velocity_interpolator_x = RegularGridInterpolator(
    #     (y, x), vx, bounds_error=False, fill_value=None
    # )
    # velocity_interpolator_y = RegularGridInterpolator(
    #     (y, x), vy, bounds_error=False, fill_value=None
    # )

    # def velocity_retina(point):
    #     return np.array(
    #         [velocity_interpolator_x(point[::-1]), velocity_interpolator_y(point[::-1])]
    #     )

    # fig, axes = plt.subplots(ncols=2, figsize=(14, 6))

    # X, Y = np.meshgrid(x, y)  # Ensure X and Y match the dimensions of sdf

    # # Add contours for the signed distance function (SDF) on both subplots
    # axes[0].contour(X, Y, sdf, levels=[0], colors="black", linewidths=1, zorder=10)
    # axes[1].contour(X, Y, sdf, levels=[0], colors="black", linewidths=1, zorder=10)

    # # Plot velocity components as filled contours
    # vxmin = np.min(vx)
    # vxmax = np.max(vx)
    # normx = mcolors.TwoSlopeNorm(vmin=vxmin, vcenter=0, vmax=vxmax)
    # vymin = np.min(vy)
    # vymax = np.max(vy)
    # normy = mcolors.TwoSlopeNorm(vmin=vymin, vcenter=0, vmax=vymax)

    # contourx = axes[0].contourf(X, Y, vx, levels=100, cmap="RdBu",norm=normx)
    # contoury = axes[1].contourf(X, Y, vy, levels=100, cmap="RdBu",norm=normy)

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
