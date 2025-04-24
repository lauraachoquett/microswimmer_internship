import numpy as np

from .Astar import astar
from .sdf import sdf_circle


def find_shortest_path_circle(start,goal,min_distance,Z):
    return  np.array(astar(start, goal, Z, min_distance))


