import numpy as np
import matplotlib.pyplot as plt
from math import atan2,pi,exp,cos
from scipy.interpolate import CubicSpline
from utils import courbures


def generate_simple_line(p_0,p_target,nb_points):
    t = np.linspace(0,1,nb_points)
    path = p_0 * t[:,None] +(1-t)[:,None]*p_target
    d = np.linalg.norm((p_target - p_0))
    return np.flip(path,axis=0),d

def generate_line_two_part(p_0,p_1,p_target,nb_points):
    t = np.linspace(0,1,nb_points)
    path_1 = p_0 * t[:,None] +(1-t)[:,None]*p_1
    path_1 = np.flip(path_1,axis=0)
    path_2 = np.flip(p_1 * t[:,None] +(1-t)[:,None]*p_target,axis=0)
    path= np.concatenate((path_1,path_2),axis=0)
    d = np.linalg.norm(p_1-p_0) + np.linalg.norm(p_target - p_0)
    return path,d

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
    d = pi * r
    return np.flip(path,axis=0),d

def generate_curve(p_0,p_target,k,nb_points): 
    t = np.linspace(0, 1, nb_points)  
    x = (1 - t) * p_0[0] + t * p_target[0]
    y = (1 - t) * p_0[1] + t * p_target[1] + k * t * (1 - t) * (p_target[0] - p_0[0])  
    return np.concatenate((x.reshape(nb_points,1),y.reshape(nb_points,1)),axis=1)

def generate_random_ondulating_path(p_0, p_target, n_points=100, max_curvature=1.0, amplitude=0.1, frequency=5):
    t = np.linspace(0, 1, n_points)
    x_base = np.linspace(p_0[0], p_target[0], n_points)
    y_base = np.linspace(p_0[1], p_target[1], n_points)
    noise = amplitude * np.sin(frequency * 2 * np.pi * t)  
    y_ondulating = y_base + noise
    cs = CubicSpline(t, np.column_stack([x_base, y_ondulating]), bc_type='natural')
    path = cs(t)
    return path

def func_k_max(A,N,f,n):
    freq = f / (8*(1 + (n /N)**(3/2)))
    return -A*(1-exp(n/N))*cos(n*freq)



def plot_path(p_0,p_target,nb_points,type='line'):
    if type=='line':
        path,_ = generate_simple_line(p_0,p_target,nb_points)
    if type=='two_lines':
        p_1 = [1/2,1]
        path = generate_line_two_part(p_0,p_1,p_target,nb_points)
    if type=='circle':
        path,_ = generate_demi_circle_path(p_0,p_target,nb_points)
    if type=='curve':
        p_0 = np.zeros(2)
        p_target = np.array([1/4,1/8])
        nb_points=100
        k=2
        path = generate_curve(p_0,p_target,k,nb_points)
    if type=='ondulating_path':
        path=generate_random_ondulating_path(p_0,p_target,nb_points,amplitude=0.8,frequency=10)
    print(np.max(courbures(path)))
    plt.close()
    plt.plot(path[:,0],path[:,1],label='path')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(p_0[0], p_0[1], color='red', label="Starting point")
    plt.scatter(p_target[0], p_target[1], color='blue', label="Ending point")
    plt.legend()
    plt.title(f"Path : {type}")
    plt.savefig(f"fig/path_{type}.png",dpi=100,bbox_inches='tight')
    
if __name__  == '__main__' : 
    p_0 = np.array([0,0])
    p_target  =np.array([2,0])
    nb_points = 200
    # plot_path(p_0,p_target,nb_points,'line')
    A=2
    N=700
    f = 4
    n_values = np.linspace(1,N+1,700,dtype=int)
    output = [func_k_max(A,N,f,n) for n in n_values]
    nb_points=700
    plt.figure(figsize=(25,10))
    plt.subplot(1,2,1)
    plt.plot(n_values,output)
    colors = plt.cm.viridis(np.linspace(0, 1, 20))
    for n in n_values:
        if n>440 and n<460:
            k= func_k_max(A,N,f,n)
            plt.scatter(n, k, color=colors[n%20]) 
            
    plt.subplot(1,2,2)
    for n in n_values:
        if n>440  and n<460:
            k= func_k_max(A,N,f,n)
            path = generate_curve(p_0,p_target,k,nb_points)
            plt.plot(path[:,0],path[:,1],label=f'k : {k:.2f}',color = colors[n%20])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(p_0[0], p_0[1],color = 'black')
    plt.scatter(p_target[0], p_target[1], color='black')
    plt.legend()
    plt.savefig('fig/smooth_curve.png',dpi=200,bbox_inches='tight')
            
            
    
    
