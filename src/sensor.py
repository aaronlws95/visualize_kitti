import numpy as np

CAM02_PARAMS = dict(
    fx = 7.215377e+02,
    fy = 7.215377e+02,
    cx = 6.095593e+02,
    cy = 1.728540e+02,
    rot = [[9.999758e-01, -5.267463e-03, -4.552439e-03],
           [5.251945e-03, 9.999804e-01, -3.413835e-03],
           [4.570332e-03, 3.389843e-03, 9.999838e-01]],
    trans = [[5.956621e-02], [2.900141e-04], [2.577209e-03]],
)

VELO_PARAMS = dict(
    rot = [[7.533745e-03, -9.999714e-01, -6.166020e-04],
           [1.480249e-02, 7.280733e-04, -9.998902e-01],
           [9.998621e-01, 7.523790e-03, 1.480755e-02]],
    trans = [[-4.069766e-03], [-7.631618e-02], [-2.717806e-01]],
)

def get_intrinsic(fx, fy, cx, cy):
    return np.asarray([[fx, 0, cx],
                       [0, fy, cy],
                       [0, 0, 1]])

def get_extrinsic(rot, trans):
    return np.hstack((rot, trans))