from generate_path import generate_simple_line
from sde import *
import time
colors = plt.cm.tab10.colors
from scipy.spatial import KDTree

def min_dist_closest_point(x,tree):
    dist, idx = tree.query(x)
    return dist, idx


def distance_to_path(tree,nb_steps,t_init,t_end,U=1,p = np.ones(2),x_0 = np.zeros(2)):
    D=0.1

    Dt = (t_end-t_init)/nb_steps

    traj = np.zeros((nb_steps,2))
    distances = np.zeros((nb_steps,2))
    traj[0] = x_0



    for n in range(nb_steps-1) : 
        traj[n+1] = solver(traj[n],U,p,Dt,D)
        distances[n],_= min_dist_closest_point(traj[n],tree)
    distances[-1],_ = min_dist_closest_point(traj[-1],tree)

    return traj,distances

def plot_distances(nb_steps,t_init,t_end,nb_points_path,nb_sims):
    p_target = np.ones(2)
    p_0 = np.zeros(2)
    fig = plt.figure(figsize=(12, 4))
    path = generate_simple_line(p_0,p_target,nb_points_path)
    tree = KDTree(path)
    for i in range(nb_sims):
        traj,distances = distance_to_path(tree,nb_steps,t_init,t_end)
        plt.subplot(1,2,1)
        plt.plot(distances,color = colors[i])
        plt.xlabel("t")
        plt.ylabel("distance")
        plt.title("Distance to the path at every time step")
        plt.subplot(1,2,2)
        indices = np.linspace(0, len(path) - 1, nb_steps).astype(int)
        path = path[indices]
        plt.plot(path[:,0],path[:,1],label ='path',linewidth=3,color = 'r')
        plt.plot(traj[:,0],traj[:,1],label='trajectory',color = colors[i])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Trajectory and path")
    plt.savefig("fig/distances_to_path.png",dpi=100,bbox_inches='tight')

if __name__ == '__main__':
    nb_steps=100
    t_init=0
    t_end=1
    nb_points_path=1000
    nb_sims=10
    start = time.time()
    plot_distances(nb_steps,t_init,t_end,nb_points_path,nb_sims)
    print("Durée:", time.time() - start)

    
