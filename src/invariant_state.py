import numpy as np
from math import atan,sin,cos

import numpy as np
from math import sin, cos, atan2

def coordinate_in_path_ref(p, dir, x):

    theta = atan2(dir[1], dir[0])
    
    x = np.array(x) - np.array(p)
    sin_th = sin(theta)
    cos_th = cos(theta)
    
    R = np.array([[cos_th, -sin_th],
                  [sin_th,  cos_th]])
    R = R.T
    
    return R @ x

def coordinate_in_global_ref(p,dir,x):
    theta = atan2(dir[1],dir[0])
    sin_th = sin(theta)
    cos_th = cos(theta)
        
    R = np.array([[cos_th, -sin_th],
                  [sin_th,  cos_th]])
    x = R @ x + p

    return x






