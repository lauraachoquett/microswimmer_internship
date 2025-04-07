import numpy as np
import matplotlib.pyplot as plt

def dW(delta_t):
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t),size=(2))

def solver(x,U,p,Dt,D,u_bg=np.zeros(2)):
    dW_t =  dW(Dt)
    return x + +u_bg+U*p*Dt+ D*dW_t

def run_sde(nb_steps,t_init,t_end,U=1,p = np.ones(2),x_0 = np.zeros(2),u_bg=np.zeros(2)):
    D=0.1

    Dt = (t_end-t_init)/nb_steps

    traj = np.zeros((nb_steps,2))
    traj[0] = x_0

    for n in range(nb_steps-1) : 
        traj[n+1] = solver(traj[n],U,p,Dt,D)
    return traj
    
def plot_simulation(num_sims,nb_steps,t_init,t_end):
    for i in range(num_sims):
        traj = run_sde(nb_steps,t_init,t_end)
        plt.plot(traj[:,0],traj[:,1],label=f'Sim : {i}')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Trajectories with {nb_steps} steps for {t_end - t_init} (time)")
    plt.legend()
    plt.savefig("fig/SDE_test.png",dpi=100,bbox_inches='tight')

if __name__== '__main__':
    num_sims=5
    nb_steps=100
    t_init = 0
    t_end=1
    plot_simulation(num_sims,nb_steps,t_init,t_end)