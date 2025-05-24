
import numpy as np


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


if __name__ == '__main__':
    n, h, vel = vel_read('data/vel.sdf')
    print(vel.shape)
    v = vel
    vx = v[:,:,:,0]
    vy = v[:,:,:,1]
    vz= v[:,:,:,2]

    n, h, sdf = sdf_read('data/retina.sdf')
    sdf = sdf[n[2]//2,:,:]
    np.save("retina2D.npy", sdf, allow_pickle=False)

    for name, arr in zip(["vx", "vy","vz"], [vx, vy,vz]):
        print(f"{name}: min={arr.min():.3e}, max={arr.max():.3e}, mean={arr.mean():.3e}, std={arr.std():.3e}")
    speed = np.sqrt(vx**2 + vy**2 +vz**2 )
    print(f"Speed: min={speed.min():.3e}, max={speed.max():.3e}, mean={speed.mean():.3e}, std={speed.std():.3e}")  
    threshold = 10 * speed.std()  # à adapter
    outliers = np.argwhere(speed > 10)
    vx = vx[speed<10]
    vy= vy[speed<10]
    vz = vz[speed<10]
        for name, arr in zip(["vx", "vy","vz"], [vx, vy,vz]):
        print(f"{name}: min={arr.min():.3e}, max={arr.max():.3e}, mean={arr.mean():.3e}, std={arr.std():.3e}")
    speed = np.sqrt(vx**2 + vy**2 +vz**2 )
    print(f"Speed: min={speed.min():.3e}, max={speed.max():.3e}, mean={speed.mean():.3e}, std={speed.std():.3e}")  
    # print(f"Nb points avec vitesse > {threshold:.2f} : {len(outliers)}")
    import matplotlib.pyplot as plt



    plt.hist(speed.flatten(), bins=100)
    plt.yscale("log")
    plt.title("Histogramme des vitesses")
    plt.xlabel("||v||")
    plt.ylabel("Fréquence (log)")
    plt.savefig('fig/hist_v')
    
    

    # # === Statistiques après filtrage ===
    # filtered_speed = np.sqrt(vx**2 + vy**2 + vz**2)
    # print(f"After filtering: max={filtered_speed.max():.3e}, min={filtered_speed.min():.3e}, mean={filtered_speed.mean():.3e}, std={filtered_speed.std():.3e}")
    # for name, arr in zip(["vx", "vy", "vz","sdf"], [vx, vy, vz,sdf]):
    #     print(f"After filtering {name}:  max={arr.max():.3e}, min={arr.min():.3e}, mean={arr.mean():.3e}, std={arr.std():.3e}")
    # plt.figure(figsize = (30,8))
    # plt.subplot(1,3,1)
    # plt.hist(vx.flatten(), bins=100)
    # plt.yscale("log")
    # plt.xlabel("vx")
    # plt.ylabel("Frequency (log)")
    # plt.yscale("log")
    # plt.subplot(1,3,2)
    # plt.hist(vy.flatten(), bins=100)
    # plt.xlabel("vy")
    # plt.ylabel("Frequency (log)")
    # plt.yscale("log")
    # plt.subplot(1,3,3)
    # plt.hist(vz.flatten(), bins=100)
    # plt.xlabel("vz")
    # plt.ylabel("Frequency (log)")
    # plt.yscale("log")
    # plt.savefig('fig/hist_v_mask')
    # plt.close()
    
    # speed = np.sqrt(vx**2 + vy**2 + vz**2)
    # plt.hist(speed.flatten(), bins=100)
    # plt.yscale("log")
    # plt.title("Histogramme des vitesses")
    # plt.xlabel("||v||")
    # plt.ylabel("Frequency (log)")
    # plt.savefig('fig/hist_v_norm_mask')
    # plt.close()