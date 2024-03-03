import argparse
from pathlib import Path
import os

import numpy as np

from vgn.detection import VGN_NO_ROS as VGN
from vgn.dataset import Dataset
from vgn.grasp import Grasp
from vgn.utils.transform import Rotation, Transform
from vgn import vis

def main(args):
    grasp_planner = VGN(Path(args.model))

    dataset = Dataset(Path(args.dataset), augment=False)
    i = np.random.randint(len(dataset))

    size, cloud, voxel_grid, (label, rotations, width), index = dataset.get_item(i)
    grasp = Grasp(Transform(Rotation.from_quat(rotations[0]), index), width)
    # vis.draw_scene(size,cloud,voxel_grid,grasp,float(label), 40.0 / 6.0)

    grasps, scores, timings = grasp_planner(voxel_grid,1)  
    print(timings)
    grasp, score = grasps[0], scores[0]
    vis.draw_scene(size, cloud, voxel_grid,grasp,score,40.0/6.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='data/models/vgn_conv.pth')
    parser.add_argument('--dataset',default='data/datasets/minimass')
    args = parser.parse_args()
    main(args)
