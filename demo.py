import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

import src.sensor as sensor
import src.utils as utils
import src.visualize as vis
from src.dataset import load_data

# Path info
base_path = Path('data')/'KITTI'
date = '2011_09_26'
drive = 5

# Setup
dataset = load_data(base_path, date, drive)
i = 22
h_fov = (-90, 90)
v_fov = (-24.9, 2.0)

# Show Image
pcl_uv, pcl_d, img, _ = dataset.get_projected_pts(i, h_fov, v_fov)
vis.show_projection(pcl_uv, pcl_d, img)

# Write Video
vis.write_to_video(dataset, h_fov, v_fov)

