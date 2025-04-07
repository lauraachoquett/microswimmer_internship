import numpy as np
import matplotlib.pyplot as plt

def generate_simple_line(p_0,p_target,nb_points):
    t = np.linspace(0,1,nb_points)
    path = p_0 * t[:,None] +(1-t)[:,None]*p_target
    return np.flip(path,axis=0)

def plot_path(p_0,p_target,nb_points,type='line'):
    if type=='line':
        path = generate_simple_line(p_0,p_target,nb_points)
    plt.plot(path[:,0],path[:,1],label='path')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Path : {type}")
    plt.savefig(f"fig/path_{type}.png",dpi=100,bbox_inches='tight')

if __name__  == '__main__' : 
    p_0 = np.zeros(2)
    p_target  =2*np.ones(2)
    nb_points = 200
    plot_path(p_0,p_target,nb_points)