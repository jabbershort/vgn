import argparse
from pathlib import Path

import numpy as np

from vgn import vis
from vgn.dataset import Dataset
from vgn.grasp import Grasp
from vgn.utils.transform import Rotation, Transform
import open3d as o3d


def main(args):

    dataset = Dataset(args.dataset, augment=args.augment)
    i = np.random.randint(len(dataset))

    cloud, voxel_grid, (label, rotations, width), index = dataset[i]
    grasp = Grasp(Transform(Rotation.from_quat(rotations[0]), index), width)

    vis.draw_scene(cloud,grasp,float(label), 40.0 / 6.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",default='data/datasets/matt' ,type=Path)
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()
    main(args)
