import numpy as np
import matplotlib.pyplot as plt
def generate_simple_line(x_0,x_target,nb_points):
    t = np.linspace(0,1,nb_points)
    return x_0 * t[:,None] +(1-t)[:,None]*x_target

def plot_path(x_0,x_target,nb_points,type='line'):
    if type=='line':
        path = generate_simple_line(x_0,x_target,nb_points)
    plt.plot(path[:,0],path[:,1],label='path')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Path : {type}")
    plt.savefig(f"fig/path_{type}.png",dpi=100,bbox_inches='tight')

if __name__  == '__main__' : 
    x_0 = np.zeros(2)
    x_target  =2*np.ones(2)
    nb_points = 200
    plot_path(x_0,x_target,nb_points)