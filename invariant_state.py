import numpy as np
from math import atan,sin,cos

import numpy as np
from math import sin, cos, atan2

def coordinate_in_path_ref(p_0, T_0, x):
    
    theta = atan2(T_0[1], T_0[0])
    
    x = np.array(x) - np.array(p_0)
    sin_th = sin(theta)
    cos_th = cos(theta)
    
    R = np.array([[cos_th, -sin_th],
                  [sin_th,  cos_th]])
    R = R.T
    
    return R @ x

def coordinate_in_global_ref(p_0,T_0,x):
    theta = atan2(T_0[1], T_0[0])
    sin_th = sin(theta)
    cos_th = cos(theta)
        
    R = np.array([[cos_th, -sin_th],
                  [sin_th,  cos_th]])
    x = R @ x + p_0

    return x






