from scipy.spatial import cKDTree as KDTree
import numpy as np


def normalize_point_cloud(pc):
    min_xyz = np.min(pc, axis=0)
    max_xyz = np.max(pc, axis=0)
    centroid = min_xyz + (max_xyz - min_xyz)/2.0
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc_normalized = pc / m
    return pc_normalized, centroid, m

def voxelized_pointcloud(input_pc, grid_points, kdtree):
    point_cloud, centroid, m = normalize_point_cloud(input_pc)
    occupancies = np.zeros(len(grid_points), dtype=np.int8)
    _, idx = kdtree.query(point_cloud)
    occupancies[idx] = 1
    compressed_occupancies = np.packbits(occupancies)
    return compressed_occupancies, centroid, m, point_cloud


def init(bb_min, bb_max, input_res):
    global kdtree, grid_points
    grid_points = create_grid_points_from_bounds(bb_min, bb_max, input_res)
    kdtree = KDTree(grid_points)
    return grid_points, kdtree

def create_grid_points_from_bounds(minimun, maximum, res):
    x = np.linspace(minimun, maximum, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))
    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list
