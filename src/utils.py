import numpy as np

def get_euclidean_distance(pts):
    return np.sqrt(pts[:, 0]**2 + pts[:, 1]**2 + pts[:, 2]**2).reshape(-1, 1)

def get_fov_range_idx(n, m, fov_range):
    return np.logical_and(np.arctan2(n, m) < (fov_range[1] * np.pi / 180), np.arctan2(n, m) > (fov_range[0] * np.pi / 180))

def normalize_pts(pts):
    return (pts - np.min(pts)) / (np.max(pts) - np.min(pts))

def filter_pts_by_fov(pts, h_fov, v_fov):
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    d = np.sqrt(x**2 + y**2 + z**2)
    h_pts_idx = get_fov_range_idx(y, x, h_fov)
    v_pts_idx = get_fov_range_idx(z, d, v_fov)
    idx = np.logical_and(h_pts_idx, v_pts_idx)
    return pts[idx]

def get_projection_outlier_idx(pts, img_shape):
    u_outliers = np.logical_or(pts[:, 0] < 0, pts[:, 0] > img_shape[1])
    v_outliers = np.logical_or(pts[:, 1] < 0, pts[:, 1] > img_shape[0])
    outliers = np.logical_or(u_outliers, v_outliers)
    return outliers

def get_2D_lidar_projection(pcl, cam_intrinsic, velo_extrinsic, h_fov, v_fov):
    filter_pcl = filter_pts_by_fov(pcl, h_fov, v_fov)
    pcl_d = get_euclidean_distance(filter_pcl)
    pcl_xyz = np.hstack((filter_pcl[:, :3], np.ones((filter_pcl.shape[0],1)))).T
    pcl_xyz = velo_extrinsic@pcl_xyz
    pcl_xyz = cam_intrinsic@pcl_xyz
    pcl_xyz = pcl_xyz.T
    pcl_xyz = pcl_xyz/pcl_xyz[:, 2, None]
    pcl_uv = pcl_xyz[:, :2]
    pcl_d = normalize_pts(pcl_d)*90

    return pcl_uv, pcl_d

