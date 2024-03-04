import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

from vgn.io import *
from vgn.perception import *
from vgn.parameters import *

def main(args):
    # create directory of new dataset
    (args.dataset / "scenes").mkdir(parents=True)

    # load setup information
    size, intrinsic, max_opening_width, finger_depth = read_setup(args.raw)
    write_setup(args.dataset,size,intrinsic,max_opening_width,finger_depth)

    assert np.isclose(size, 6.0 * finger_depth)
    voxel_size = size / NetworkParameters.resolution

    # create df
    df = read_df(args.raw)
    df["x"] /= voxel_size
    df["y"] /= voxel_size
    df["z"] /= voxel_size
    df["width"] /= voxel_size
    df = df.rename(columns={"x": "i", "y": "j", "z": "k"})

    # Skim off results outside of voxel grid.
    df.drop(df[df["i"] > 39].index,inplace=True)
    df.drop(df[df["i"] < 0].index,inplace=True)
    df.drop(df[df["j"] > 39].index,inplace=True)
    df.drop(df[df["j"] < 0].index,inplace=True)
    df.drop(df[df["k"] > 39].index,inplace=True)
    df.drop(df[df["k"] < 0].index,inplace=True)
    
    write_df(df, args.dataset)

    # create tsdfs
    for f in tqdm(list((args.raw / "scenes").iterdir())):
        if f.suffix != ".npz":
            continue
        depth_imgs, extrinsics = read_sensor_data(args.raw, f.stem)
        # write_sensor_data_scene(args.dataset,f.stem,depth_imgs,extrinsics)
        tsdf = create_tsdf(size, NetworkParameters.resolution, depth_imgs, intrinsic, extrinsics)
        grid = tsdf.get_grid()
        # write_voxel_grid(args.dataset, f.stem, grid)
        write_combined(args.dataset, f.stem, depth_imgs, extrinsics, grid)

# 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    main(args)
