import models.local_model as model
from models.generation import Generator
import torch
import configs.config as cfg_loader
import os
import numpy as np
from models.utils import init, voxelized_pointcloud

def rotate_mat(radian):
    rot_matrix = np.array([[np.cos(radian), 0, -np.sin(radian)],
                          [0, 1, 0],
                          [np.sin(radian), 0, np.cos(radian)]])
    return rot_matrix

def gen_iterator(cfg, datapath, out_path, gen):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print(out_path)
    grid_points, kdtree = init(cfg.bb_min, cfg.bb_max, cfg.input_res)
    for tempFile in os.listdir(datapath):
        pc_item_path = os.path.join(datapath, tempFile)
        temp_pc = np.loadtxt(pc_item_path)
        print(f"Processing {tempFile} with {len(temp_pc)} points")
        compressed_occupancies, centroid, m, nor_pc = voxelized_pointcloud(temp_pc, grid_points, kdtree)
        canonical_pointcloud, duration = gen.generate_point_cloud(compressed_occupancies, num_steps=5)
        rot_matrix = rotate_mat(90*np.pi/180)
        canonical_pointcloud = np.matmul(canonical_pointcloud, rot_matrix)
        canonical_pointcloud = canonical_pointcloud * np.array([1,1,-1])
        canonical_pointcloud = canonical_pointcloud * m + centroid
        original_name = os.path.splitext(tempFile)[0]
        np.savetxt(os.path.join(out_path, f"{original_name}_Canonical.txt"), canonical_pointcloud)

if __name__ == "__main__":
    cfg = cfg_loader.get_config()
    device = torch.device("cuda:0")
    net = model.INR()
    gen = Generator(net, cfg.exp_dir, device=device)
    out_path = f'./outputs/'
    datapath = './testData/'
    gen_iterator(cfg, datapath, out_path, gen=gen)
