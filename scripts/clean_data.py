import argparse
from vgn import vis
from vgn.io import *
from vgn.perception import *
from vgn.utils.transform import Rotation,Transform
from pathlib import Path
import matplotlib.pyplot as plt


def main(args):
#    root = Path('data/raw/foo')
    df = read_df(args.root)

    positives = df[df["label"] == 1]
    negatives = df[df["label"] == 0]

    print("Number of samples:", len(df.index))
    print("Number of positives:", len(positives.index))
    print("Number of negatives:", len(negatives.index))

    size, intrinsic, _, finger_depth = read_setup(args.root)

    i = np.random.randint(len(df.index))
    scene_id, grasp, label = read_grasp(df, i)
    depth_imgs, extrinsics = read_sensor_data(args.root, scene_id)

    angles = np.empty(len(positives.index))
    for i, index in enumerate(positives.index):
        approach = Rotation.from_quat(df.loc[index, "qx":"qw"].to_numpy()).as_matrix()[:,2]
        angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))
        angles[i] = np.rad2deg(angle)        

    if args.vis:
        plt.hist(angles, bins=30)
        plt.xlabel("Angle [deg]")
        plt.ylabel("Count")
        plt.show()

    df = read_df(args.root)
    df.drop(df[df["x"] < 0.02].index, inplace=True)
    df.drop(df[df["y"] < 0.02].index, inplace=True)
    df.drop(df[df["z"] < 0.02].index, inplace=True)
    df.drop(df[df["x"] > 0.28].index, inplace=True)
    df.drop(df[df["y"] > 0.28].index, inplace=True)
    df.drop(df[df["z"] > 0.28].index, inplace=True)
    write_df(df, args.root)

    df = read_df(args.root)
    scenes = df["scene_id"].values
    for f in (args.root / "scenes").iterdir():
        if f.suffix == ".npz" and f.stem not in scenes:
            print("Removed", f)
            f.unlink()

    df = read_df(args.root)
    positives = df[df["label"] == 1]
    negatives = df[df["label"] == 0]
    i = np.random.choice(negatives.index, len(negatives.index) - len(positives.index), replace=False)
    df = df.drop(i)

    write_df(df, args.root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument('--vis', action="store_true")
    args = parser.parse_args()
    main(args)
