import numpy as np
import matplotlib.pyplot as plt
from math import atan2
def generate_simple_line(p_0,p_target,nb_points):
    t = np.linspace(0,1,nb_points)
    path = p_0 * t[:,None] +(1-t)[:,None]*p_target
    return np.flip(path,axis=0)

def generate_line_two_part(p_0,p_1,p_target,nb_points):
    t = np.linspace(0,1,nb_points)
    path_1 = p_0 * t[:,None] +(1-t)[:,None]*p_1
    path_1 = np.flip(path_1,axis=0)
    path_2 = np.flip(p_1 * t[:,None] +(1-t)[:,None]*p_target,axis=0)
    path= np.concatenate((path_1,path_2),axis=0)
    return path

def generate_demi_circle_path(p_0, p_target, nb_points):
    p_0 = np.array(p_0)
    p_target = np.array(p_target)

    dir_vec = p_target - p_0
    length = np.linalg.norm(dir_vec)
    r = length / 2
    center = p_0 + dir_vec / 2

    alpha = np.arctan2(dir_vec[1], dir_vec[0])
    thetas = np.linspace(0, np.pi, nb_points)
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    circle_points = np.stack([x, y], axis=1)
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]])
    rotated_points = circle_points @ R.T
    path = center + rotated_points

    return np.flip(path,axis=0)

def plot_path(p_0,p_target,nb_points,type='line'):
    if type=='line':
        path = generate_simple_line(p_0,p_target,nb_points)
    if type=='two_lines':
        p_1 = [1/2,1]
        path = generate_line_two_part(p_0,p_1,p_target,nb_points)
    if type=='circle':
        path = generate_demi_circle_path(p_0,p_target,nb_points)
    plt.plot(path[:,0],path[:,1],label='path')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Path : {type}")
    plt.savefig(f"fig/path_{type}.png",dpi=100,bbox_inches='tight')

if __name__  == '__main__' : 
    p_0 = np.array([-2,-1])
    p_target  =np.array([2,1])
    nb_points = 200
    plot_path(p_0,p_target,nb_points,'circle')