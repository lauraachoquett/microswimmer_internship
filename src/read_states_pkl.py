import os
import pickle
import pickletools
import numpy as np
import json 
import matplotlib.pyplot as plt
from itertools import chain
from src.data_loader import load_sdf_from_csv, vel_read, load_sim_sdf
from src.Astar_ani import astar_anisotropic, compute_v, contour_2D

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import zipfile

# Load trajectory data

def create_data():
       file = 'grid_search/8/result_evaluation_retina__traj.pkl'
       file_bis = 'grid_search/9/result_evaluation_retina__traj.pkl'
       with open(file, "rb") as f:
              successful_trajectories_1 = pickle.load(f)
       with open(file_bis, "rb") as f:
              successful_trajectories_bis = pickle.load(f)

       successful_trajectories = successful_trajectories_1 + successful_trajectories_bis

       print(len(successful_trajectories))
       # Configuration de la simulation
       ratio = 5
       sdf_func, velocity_retina, x_phys, y_phys, physical_width, physical_height, scale = load_sim_sdf(ratio)
       X, Y = np.meshgrid(x_phys, y_phys)
       obstacle_contour = contour_2D(sdf_func, X, Y, scale)

       dir = './export_data'
       os.makedirs(dir,exist_ok=True)
       trajectories_dict = {
       f"traj_{i}": traj.tolist() for i, traj in enumerate(successful_trajectories)
       }

       # Structure finale
       data = {
       "trajectories": trajectories_dict
       }

       with open(os.path.join(dir,"trajectories.json"), "w") as f:
              json.dump(data, f, indent=2)

       print("Saved trajectories.json")
       df_obs = pd.DataFrame(obstacle_contour, columns=["x", "y"])
       df_obs.to_csv(os.path.join(dir,"obstacle.csv"), index=False)

       print("Saved obstacle.csv")


def create_fig(successful_trajectories,obstacle_contour):
       fig, ax = plt.subplots(figsize=(12, 10))

       # Obstacles (réseau capillaire)
       ax.scatter(obstacle_contour[:, 0], obstacle_contour[:, 1], 
              color="black", s=0.3, alpha=0.8, label="Réseau capillaire",rasterized=True)

       # Trajectoires
       alpha_traj = 0.2 
       color_blue = '#0072B2'
       for i, trajectory in enumerate(successful_trajectories):
              traj = np.array(trajectory)
       
              ax.plot(traj[:, 0], traj[:, 1], alpha=alpha_traj, color=color_blue, rasterized=True,linewidth=2)
              
              # Points de départ et d'arrivée
              ax.plot(traj[0, 0], traj[0, 1], marker='o', color='orange', 
                     markersize=7, alpha=0.6,rasterized=True)
              ax.plot(traj[-1, 0], traj[-1, 1], marker='x', color='red', 
                     markersize=6, alpha=0.6,rasterized=True)

       ax.set_aspect('equal')
       ax.set_axis_off()
       ax.grid(False)


       from matplotlib.lines import Line2D
       legend_elements = [

       Line2D([0], [0], color=color_blue, alpha=0.7, linewidth=2, 
              label=f'Successful Trajectories'),

       Line2D([0], [0], color='red', marker='x', linestyle='None', 
              markersize=10, label='Target Points'),
       Line2D([0], [0], color='orange', marker='o', linestyle='None', 
              markersize=7, label='Starting Point')
       ]
       ax.legend(handles=legend_elements, loc='upper left', fontsize=10)


       plt.tight_layout()
       plt.savefig('fig/trajectories_density.png',dpi=400,bbox_inches='tight')
       plt.savefig('fig/trajectories_density.pdf',dpi=400,bbox_inches='tight')



def load_data(dir):
       with open(os.path.join(dir,"trajectories.json"),'r') as f:
              data = json.load(f)
              
       successful_trajectories = [np.array(points) for points in data["trajectories"].values()]

       df_obs = pd.read_csv(os.path.join(dir,'obstacle.csv'))
       obstacle_contour = df_obs.to_numpy()

       return successful_trajectories,obstacle_contour

if __name__=='__main__':
       
       zip_path = "data_trajectories.zip"
       extract_dir = "./"
       with zipfile.ZipFile(zip_path, "r") as zip_ref:
              zip_ref.extractall(extract_dir)
              
       successful_trajectories,obstacle_contour = load_data('./data_trajectories')
       create_fig(successful_trajectories,obstacle_contour)