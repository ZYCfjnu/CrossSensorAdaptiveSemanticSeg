import configargparse
import numpy as np
import os

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, default='configs/dataset.txt',
                        help='config file path')
    parser.add_argument("--exp_name", type=str, default=None,
                        help='Experiment name, used as folder name for the experiment. If left blank, a \
                         name will be auto generated based on the configuration settings.')
    parser.add_argument("--data_dir", type=str,
                        help='input data directory')
    parser.add_argument("--exp_dir", type=str,
                        help='Path to read and write the checkpoints')
    parser.add_argument("--sample_std_dev", action='append', type=float,
                        help='Standard deviations of gaussian samples. \
                Used for displacing surface points to sample the distance field.')
    parser.add_argument("--sample_ratio", action='append', type=float,
                        help='Ratio of standard deviations for samples used for training. \
                Needs to have the same len as sample_std with floats between 0-1 \
                and summing to 1.')
    parser.add_argument("--input_res", type=int, default=256,
                        help='Training and testing shapes are normalized to be in a common bounding box.\
                             This value defines the max value in x,y and z for the bounding box.')
    parser.add_argument("--num_cpus", type=int, default=-1,
                        help='Number of cpu cores to use for running the script. \
            Default is -1, that is, using all available cpus.')
    parser.add_argument("--batch_size", type=int, default=1,
                        help='Number of objects provided to the network in one batch during training.\
                            Influences training speed (larger batches result in shorter epochs) but also GPU \
                             memory usage (higher values need more memory). Needs to be balanced with \
                             num_sample_points_training')
    parser.add_argument("--num_epochs", type=int, default=100,
                        help='Stopping citron for duration of training. Model converges much earlier: model convergence\
                         can be checked via tensorboard and is logged within the experiment folder.')
    parser.add_argument("--lr", type=float, default=1e-6,
                        help='Learning rate used during training.')
    parser.add_argument("--bb_min", default=-1, type=float,
                        help='Training and testing shapes are normalized to be in a common bounding box.\
                             This value defines the min value in x,y and z for the bounding box.')
    parser.add_argument("--bb_max", default=1, type=float,
                        help='Training and testing shapes are normalized to be in a common bounding box.\
                             This value defines the max value in x,y and z for the bounding box.')
    return parser

def get_config():
    parser = config_parser()
    cfg = parser.parse_args()
    cfg.sample_ratio = np.array(cfg.sample_ratio)
    cfg.sample_std_dev = np.array(cfg.sample_std_dev)
    assert np.sum(cfg.sample_ratio) == 1
    assert np.any(cfg.sample_ratio < 0) == False
    assert len(cfg.sample_ratio) == len(cfg.sample_std_dev)
    return cfg