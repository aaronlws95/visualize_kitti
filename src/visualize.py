import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from moviepy.editor import *
import imageio
import sys

from src.dataset import load_data
import src.sensor
import src.utils

def show_img(img):
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.show()

def show_pcl(pcl):
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    velo_range = range(0, pcl.shape[0], 100)
    ax.scatter(pcl[velo_range, 0],
                pcl[velo_range, 1],
                pcl[velo_range, 2],
                c=pcl[velo_range, 3],
                cmap='gray')
    plt.show()

def get_projected_img(pts, dist, img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for i in range(pts.shape[0]):
        cv2.circle(hsv_img, (int(pts[i, 0]), int(pts[i, 1])), radius=2, color=(int(dist[i]), 255, 255), thickness=-1)

    projection = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    return projection

def show_projection(pts, dist, img):
    projection = get_projected_img(pts, dist, img)
    show_img(projection)

def write_to_video(dataset, h_fov, v_fov, save_path=Path('media')):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    img = dataset.load_image(0)
    vid_size = (img.shape[1], img.shape[0])
    save_path = str(save_path/'projection')
    vid = cv2.VideoWriter(save_path + '.avi', fourcc, 25.0, vid_size)

    num_imgs = len(list(dataset.img_path.glob('*')))

    print('Writing video to {}'.format(save_path))
    for i in tqdm(range(num_imgs)):
        pcl_uv, pcl_d, img, _ = dataset.get_projected_pts(i, h_fov, v_fov)
        projection = get_projected_img(pcl_uv, pcl_d, img)
        vid.write(projection)

    convert_to_gif(save_path)

def convert_to_gif(save_path):
    """https://gist.github.com/michaelosthege/cd3e0c3c556b70a79deba6855deb2cc8"""
    input_path = save_path + '.avi'
    output_path = save_path + '.gif'

    reader = imageio.get_reader(input_path)
    fps = reader.get_meta_data()['fps']

    print('Writing GIF to {}'.format(output_path))

    writer = imageio.get_writer(output_path, fps=fps)
    for i, im in enumerate(reader):
        sys.stdout.write('\rframe {0}'.format(i))
        sys.stdout.flush()
        writer.append_data(im)

    print('\r\nFinalizing...')
    writer.close()
    print('Done.')


