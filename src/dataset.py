from pathlib import Path
import cv2
import numpy as np
import src.sensor as sensor
import src.utils as utils

class load_data:
    def __init__(self, base_path, date, drive):
        self.calib_path = Path(base_path)/date/'calib'
        self.img_path = Path(base_path)/date/(date + '_drive_{:04d}_sync'.format(drive))/'image_02'/'data'
        self.lidar_path = Path(base_path)/date/(date + '_drive_{:04d}_sync'.format(drive))/'velodyne_points'/'data'
        self.CAM02_PARAMS = sensor.CAM02_PARAMS
        self.VELO_PARAMS = sensor.VELO_PARAMS
        self.cam_intrinsic = sensor.get_intrinsic(self.CAM02_PARAMS['fx'], self.CAM02_PARAMS['fy'], self.CAM02_PARAMS['cx'], self.CAM02_PARAMS['cy'])
        self.velo_extrinsic = sensor.get_extrinsic(self.VELO_PARAMS['rot'], self.VELO_PARAMS['trans'])

    def load_image(self, index):
        return cv2.imread(str(self.img_path/'{:010d}.png'.format(index)))

    def load_lidar(self, index):
        return np.fromfile(str(self.lidar_path/'{:010d}.bin'.format(index)), dtype=np.float32).reshape(-1, 4)

    def get_projected_pts(self, index, h_fov, v_fov):
        img = self.load_image(index)
        pcl = self.load_lidar(index)
        pcl_uv, pcl_d = utils.get_2D_lidar_projection(pcl, self.cam_intrinsic, self.velo_extrinsic, h_fov, v_fov)
        outliers = utils.get_projection_outlier_idx(pcl_uv, img.shape)
        pcl_uv = pcl_uv[~outliers]
        pcl_d = pcl_d[~outliers]

        return pcl_uv, pcl_d, img, pcl