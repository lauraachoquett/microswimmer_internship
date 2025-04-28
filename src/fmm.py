import numpy as np
import matplotlib.pyplot as plt
import skfmm
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from .sdf import sdf_circle
import seaborn as sns
from .data_load import load_sdf_from_csv,vel_read
import os
def compute_fmm_path(start_point, goal_point, sdf_function, x,y,B,flow_field=None, grid_size=(200, 200), domain_size=(1.0, 1.0)):
    """
    Calcule un chemin optimal en utilisant Fast Marching Method
    
    Arguments:
        start_point: Tuple (x, y) du point de départ
        goal_point: Tuple (x, y) du point d'arrivée
        sdf_function: Fonction qui retourne la SDF en chaque point (x, y)
        flow_field: Fonction qui retourne le vecteur d'écoulement (vx, vy) en chaque point, None si pas d'écoulement
        grid_size: Dimensions de la grille discrétisée
        domain_size: Dimensions physiques du domaine
        
    Returns:
        path: Liste de points [(x, y)] représentant le chemin optimal
    """

    X, Y = np.meshgrid(x, y)
    save_path_phi = 'data/phi/'
    if os.path.exists(save_path_phi):
        phi = np.load(os.path.join(save_path_phi,'phi.npy'))
    else :
        phi = np.zeros((len(y), len(x)))
        for i in range(len(x)):
            for j in range(len(y)):
                a = sdf_function((x[i], y[j]))
                phi[j, i] = a
        phi =3* phi / np.max(np.abs(phi))
        os.makedirs(save_path_phi,exist_ok=True)
        np.save(os.path.join(save_path_phi,'phi.npy'),phi)
    print("SDF computed")
    if sdf_function(start_point)<0 or sdf_function(goal_point)<0:
        print("Invalid starting point or target point")
    speed = 1.0 / (1.0 + np.exp(-B * phi))  
    speed = np.clip(speed, 0.001, 1.0) 
    save_path_flow = 'data/velocity_flow/'
    if flow_field is not None:
        if os.path.exists(save_path_flow):
            flow_strength = np.load(os.path.join(save_path_flow,'flow_strength.npy'))
            flow_direction_x = np.load(os.path.join(save_path_flow,'flow_direction_x.npy'))
            flow_direction_y = np.load(os.path.join(save_path_flow,'flow_direction_y.npy'))
        else : 
            flow_strength = np.zeros((len(y), len(x)))
            flow_direction_x = np.zeros((len(y), len(x)))
            flow_direction_y = np.zeros((len(y), len(x)))
            
            for i in range(len(x)):
                for j in range(len(y)):
                    vx, vy = flow_field((x[i], y[j]))
                    magnitude = np.sqrt(vx**2 + vy**2)
                    if magnitude > 0:
                        flow_strength[j, i] = magnitude
                        flow_direction_x[j, i] = vx / magnitude
                        flow_direction_y[j, i] = vy / magnitude
            os.makedirs(save_path_flow,exist_ok=True)
            np.save(os.path.join(save_path_flow,'flow_strength.npy'),flow_strength)
            np.save(os.path.join(save_path_flow,'flow_direction_x.npy'),flow_direction_x)
            np.save(os.path.join(save_path_flow,'flow_direction_y.npy'),flow_direction_y)
            
            
        flow_factor = 1
        speed = speed * (1.0 + flow_factor * flow_strength)
        print("Flow computed")
    mask = np.ones_like(phi, dtype=bool)
    i_goal = np.argmin(np.abs(x - goal_point[0]))
    j_goal = np.argmin(np.abs(y - goal_point[1]))
    mask[j_goal, i_goal] = False
    
    travel_time = skfmm.travel_time(mask, speed, dx=domain_size[0]/grid_size[0])
    
    travel_time = gaussian_filter(travel_time, sigma=1.0)
    
    path = []
    current = start_point
    path.append(current)
    
    travel_time_interp = RegularGridInterpolator((y, x), travel_time, bounds_error=False, fill_value=None)
    
    step_size = min(domain_size) / 100
    max_iterations = 1000
    convergence_threshold = step_size / 2
    
    for _ in range(max_iterations):
        eps = min(domain_size) / 1000
        dx_points = [(current[0] + eps, current[1]), (current[0] - eps, current[1])]
        dy_points = [(current[0], current[1] + eps), (current[0], current[1] - eps)]
        
        dx_values = [travel_time_interp(p[::-1]) for p in dx_points]  # Note: inversé car travel_time_interp attend (y, x)
        dy_values = [travel_time_interp(p[::-1]) for p in dy_points]
        
        gradient_x = (dx_values[0] - dx_values[1]) / (2 * eps)
        gradient_y = (dy_values[0] - dy_values[1]) / (2 * eps)
        
        # Normaliser le gradient
        gradient_norm = np.sqrt(gradient_x**2 + gradient_y**2)
        if gradient_norm < 1e-10:
            break
            
        gradient_x /= gradient_norm
        gradient_y /= gradient_norm
        
        next_x = current[0] - step_size * gradient_x
        next_y = current[1] - step_size * gradient_y
        next_point = (next_x, next_y)
        
        distance_to_goal = np.sqrt((next_point[0] - goal_point[0])**2 + (next_point[1] - goal_point[1])**2)
        if distance_to_goal < convergence_threshold:
            path.append(goal_point)
            break
            
        path.append(next_point)
        current = next_point
    
    return path, travel_time, (x, y)

def visualize_results(path, travel_time, grid_info, sdf_function, flow_field=None, grid_size=(200, 200), domain_size=(1.0, 1.0)):
    """
    Visualise le chemin trouvé, la carte de temps, la SDF et l'écoulement
    """
    x, y = grid_info
    X, Y = np.meshgrid(x, y)
    
    phi = np.zeros((len(y), len(x)))
    for i in range(len(x)):
        for j in range(len(y)):
            a = sdf_function((x[i], y[j]))
            phi[j, i] = a
    phi = phi / np.max(np.abs(phi))

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    
    contour = ax[0].contourf(X, Y, travel_time, 50, cmap='viridis')
    fig.colorbar(contour, ax=ax[0], label='Travel time')
    
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    ax[0].plot(path_x, path_y, 'r-', linewidth=2)
    ax[0].set_aspect('equal')
    ax[0].set_title('Map of travel time')
    
    contour_sdf = ax[1].contourf(X, Y, phi, levels=100, cmap="viridis")
    plt.colorbar(contour_sdf, label="Signed Distance")
    ax[1].contour(X, Y, phi, levels=[0], colors='k', linewidths=2)

    ax[1].plot(path_x, path_y, 'r-', linewidth=2)
    ax[1].set_aspect('equal')
    ax[1].set_title('SDF with path')
    
    if flow_field is not None:
        flow_x = np.zeros((len(y), len(x)))
        flow_y = np.zeros((len(y), len(x)))
        
        for i in range(len(x)):
            for j in range(len(y)):
                if phi[j, i] > 0:  
                    vx, vy = flow_field((x[i], y[j]))
                    flow_x[j, i] = vx
                    flow_y[j, i] = vy
        
        step = 5
        ax[1].quiver(X[::step, ::step], Y[::step, ::step], 
                   flow_x[::step, ::step], flow_y[::step, ::step], 
                   color='b', scale=50, alpha=0.7)
    
    ax[0].plot(path_x[0], path_y[0], 'ro', markersize=4)
    ax[0].plot(path_x[-1], path_y[-1], 'ro', markersize=4)
    ax[1].plot(path_x[0], path_y[0], 'ro', markersize=4)
    ax[1].plot(path_x[-1], path_y[-1], 'ro', markersize=4)
    

    
    plt.tight_layout()
    
    return fig

def sdf_many_circle(point):
    centers = [(1/3, 1/3), (3/4, 3/4)]
    radius = 1/5
    min_distance = 0.1
    distances = np.zeros(len(centers))
    for i, center in enumerate(centers):
        distances[i] = np.linalg.norm(np.array(center) - np.array(point))

    id = np.argmin(distances)
    center = centers[id]
    return sdf_circle(point, center, radius)

def simple_flow(point):
    return (1.0, 1.0)  


    
def plot_different_B():
    fig, ax = plt.subplots(figsize=(15, 7))
    start_point = (0.05, 0.5)
    goal_point = (0.95, 0.5)
    B_values = np.linspace(3,10,3,dtype='int')
    palette = sns.color_palette()
    grid_size=(200, 200)
    domain_size=(1.0, 1.0)
    x = np.linspace(0, domain_size[0], grid_size[0])
    y = np.linspace(0, domain_size[1], grid_size[1])
    for id,B in enumerate(B_values):
        path, travel_time, grid_info = compute_fmm_path(start_point, goal_point, sdf_many_circle,x,y,B,None)
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, '-', linewidth=2,color=palette[id+2],label=f'B : {B}')
    ax.set_aspect('equal')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    x, y = grid_info
    X, Y = np.meshgrid(x, y)
    phi = np.zeros((len(y), len(x)))
    for i in range(len(x)):
        for j in range(len(y)):
            phi[j, i] = sdf_many_circle((x[i], y[j]))
    contour_sdf = ax.contourf(X, Y, phi, levels=100, cmap="viridis")
    cbar = plt.colorbar(contour_sdf, ax=ax, pad=0.1, label="Signed Distance")
    cbar.ax.tick_params(labelsize=10)
    ax.contour(X, Y, phi, levels=[0], colors='k', linewidths=2)
    ax.set_title('SDF with path')
    plt.savefig('fig/fmm_B.png',dpi=200,bbox_inches='tight')
    plt.close()
    
def circle_path():
    fig, ax = plt.subplots(figsize=(15, 7))  
    start_point = (0.05, 0.5)
    goal_point = (0.95, 0.5)
    domain_size = (1.0,1.0)
    grid_size=(200,200)
    x = np.linspace(0,domain_size[0],grid_size[0])
    y = np.linspace(0,domain_size[1],grid_size[1])
    B = 5
    palette = sns.color_palette()
    path, travel_time, grid_info = compute_fmm_path(start_point, goal_point, sdf_many_circle,x,y,B,None,grid_size=grid_size,domain_size=domain_size)
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    ax.plot(path_x, path_y, '-', linewidth=2,color=palette[0],label=f'B : {B}')
    ax.set_aspect('equal')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    x, y = grid_info
    X, Y = np.meshgrid(x, y)
    phi = np.zeros((len(y), len(x)))
    for i in range(len(x)):
        for j in range(len(y)):
            phi[j, i] = sdf_many_circle((x[i], y[j]))
    contour_sdf = ax.contourf(X, Y, phi, levels=100, cmap="coolwarm")
    cbar = plt.colorbar(contour_sdf, ax=ax, pad=0.1, label="Signed Distance")
    cbar.ax.tick_params(labelsize=10)
    ax.contour(X, Y, phi, levels=[0], colors='k', linewidths=2)
    ax.set_title('SDF with path')
    plt.savefig('fig/fmm_two_circles.png',dpi=200,bbox_inches='tight')
    plt.close()
    
    
def retina_path():
    scale = 3
    domain_size =(1*scale,1*scale)
 
    x,y,N,h,sdf = load_sdf_from_csv(domain_size)
    grid_size = (N[0],N[1])
    X, Y = np.meshgrid(x, y)
    start_point = (0.98*scale, 0.3*scale)
    goal_point = (0.44*scale, 0.55*scale)
    sdf_interpolator = RegularGridInterpolator((y, x), sdf, bounds_error=False, fill_value=None)

    def sdf_function(point):
        return sdf_interpolator(point[::-1]) 

 
    path_vel = 'data/vel.sdf'
    N, h, vel = vel_read(path_vel)
    v = vel[N[2]//2,:,:,0:2]
    vx = v[:,:,0]
    vy = v[:,:,1]
    velocity_interpolator_x = RegularGridInterpolator((y, x), vx, bounds_error=False, fill_value=None)
    velocity_interpolator_y = RegularGridInterpolator((y, x), vy, bounds_error=False, fill_value=None)

    def velocity_retina(point):
        return np.array([velocity_interpolator_x(point[::-1]),velocity_interpolator_y(point[::-1])])
        
    path, travel_time, grid_info = compute_fmm_path(start_point, goal_point, sdf_function, x, y, B=20,flow_field=velocity_retina,grid_size=grid_size,domain_size=domain_size)

    
    fig = visualize_results(path, travel_time, (x, y), sdf_function, flow_field=velocity_retina)
    plt.savefig('fig/fmm_retina_velocity_2.png',dpi=200,bbox_inches='tight')
    plt.close(fig)
    
def main():
    retina_path()

    # fig = visualize_results(path, travel_time, grid_info, sdf_many_circle, None)

if __name__ == "__main__":
    main()