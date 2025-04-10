import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, atan2,sqrt

def dW(delta_t):
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t),size=(2))

def solver(x,U,p,Dt,D,u_bg=np.zeros(2)):
    dW_t =  dW(Dt)
    return x + u_bg*Dt+ U*p*Dt+ sqrt(D)*dW_t


def rankine_vortex(pos, a, center, cir):
    x, y = pos
    cx, cy = center
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx**2 + dy**2)
    if r == 0:
        return np.array([0.0, 0.0])  
    v_r = 0
    if r <= a:
        v_theta = cir / (2 * np.pi) * (r / a**2)
    else:
        v_theta = cir / (2 * np.pi) * (1 / r)
    alpha = np.arctan2(dy, dx)
    v_x = -v_theta * np.sin(alpha)
    v_y = v_theta * np.cos(alpha)
    return np.array([v_x, v_y])
    

def uniform_velocity(dir, norm):
    return dir*norm


def run_sde(nb_steps,t_init,t_end,U=1,D=0.1,p = np.ones(2),x_0 = np.zeros(2)):
    maximum_curvature = 40
    l = 1/maximum_curvature
    Dt_action = 1/maximum_curvature
    Dt = Dt_action*5
    D = Dt_action*4
    print(D)
    print(sqrt(D))
    traj = np.zeros((nb_steps,2))
    traj[0] = x_0
    dir = np.array([0,1])/sqrt(2)
    norm = 0.3
    a = 0.25
    center = (0.5, 0.5)
    circulation = 2
    for n in range(nb_steps-1) : 
        u_bg = rankine_vortex(traj[n],a,center,circulation)
        traj[n+1] = solver(traj[n],U,p,Dt,D)
    return traj
    
def plot_simulation(num_sims,nb_steps,t_init,t_end):
    u_bg = np.array([0,1])*0.5

    for i in range(num_sims):
        traj = run_sde(nb_steps,t_init,t_end)
        plt.plot(traj[:,0],traj[:,1],label=f'Sim : {i}')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Trajectories with {nb_steps} steps for {t_end - t_init} (time) ")
    plt.legend()
    plt.savefig("fig/SDE_test.png",dpi=100,bbox_inches='tight')

def plot_background_velocity(type,x_bound,y_bound,a=0.25,center= (0.5, 0.5),circulation = 0.8,dir=np.zeros(2),norm=0.):
    x = np.linspace(x_bound[0], x_bound[1], 10)
    y = np.linspace(y_bound[0], y_bound[1], 10)
    X, Y = np.meshgrid(x, y)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if type == 'rankine':
                v = rankine_vortex((X[i, j], Y[i, j]), a, center, circulation)
                U[i, j] = v[0]
                V[i, j] = v[1]
            if type == 'uniform':
                v = uniform_velocity(dir,norm)
                U[i, j] = v[0]
                V[i, j] = v[1]
    plt.quiver(X, Y, U, V, scale=15, width=0.002, color='gray')
    if type =='rankine':
        plt.scatter(center[0],center[1],marker='*')
    plt.xlabel('x')
    plt.ylabel('y')

def brownian(nb_sim):
    x_final=np.zeros(nb_sim)
    for i in range(nb_sim):
        traj = run_sde(100,0,1,0,0.5)
        x_final[i]=np.linalg.norm(traj[-1])**2
    print(np.mean(x_final)/2)
    

if __name__== '__main__':
    #plot_background_velocity()
    #plt.savefig(f'fig/vorticity_field_{type}.png',dpi=100,bbox_inches='tight')
    #plt.close()
    num_sims=5
    nb_steps=100
    t_init = 0
    t_end=1
    plot_simulation(num_sims,nb_steps,t_init,t_end)

