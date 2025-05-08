import numpy as np

from src.Astar import astar
from src.sdf import sdf_circle


def find_shortest_path(start, goal, min_distance, Z):
    return np.array(astar(start, goal, Z, min_distance))
