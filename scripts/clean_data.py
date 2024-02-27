from vgn import vis
from vgn.io import *
from vgn.perception import *
from vgn.utils.transform import Rotation,Transform
from pathlib import Path
import matplotlib.pyplot as plt

root = Path('data/raw/foo')
df = read_df(root)

positives = df[df["label"] == 1]
negatives = df[df["label"] == 0]

print("Number of samples:", len(df.index))
print("Number of positives:", len(positives.index))
print("Number of negatives:", len(negatives.index))

size, intrinsic, _, finger_depth = read_setup(root)

i = np.random.randint(len(df.index))
scene_id, grasp, label = read_grasp(df, i)
depth_imgs, extrinsics = read_sensor_data(root, scene_id)

tsdf = create_tsdf(size, 40, depth_imgs, intrinsic, extrinsics)
tsdf_grid = tsdf.get_grid()
cloud = tsdf.get_cloud()

geom = []
grid = vis.generate_tsdf(tsdf_grid,1)
geom.append(grid)
cloud = cloud.scale(40/size,[0,0,0])
cloud.translate([0,0,2])
geom.append(cloud)
grip = vis.generate_grasp(grasp,label,finger_depth)
geom.append(grip)
o3d.visualization.draw_geometries(geom)

angles = np.empty(len(positives.index))
for i, index in enumerate(positives.index):
    approach = Rotation.from_quat(df.loc[index, "qx":"qw"].to_numpy()).as_matrix()[:,2]
    angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))
    angles[i] = np.rad2deg(angle)        

plt.hist(angles, bins=30)
plt.xlabel("Angle [deg]")
plt.ylabel("Count")
plt.show()

df = read_df(root)
df.drop(df[df["x"] < 0.02].index, inplace=True)
df.drop(df[df["y"] < 0.02].index, inplace=True)
df.drop(df[df["z"] < 0.02].index, inplace=True)
df.drop(df[df["x"] > 0.28].index, inplace=True)
df.drop(df[df["y"] > 0.28].index, inplace=True)
df.drop(df[df["z"] > 0.28].index, inplace=True)
write_df(df, root)

df = read_df(root)
scenes = df["scene_id"].values
for f in (root / "scenes").iterdir():
    if f.suffix == ".npz" and f.stem not in scenes:
        print("Removed", f)
        f.unlink()

df = read_df(root)
positives = df[df["label"] == 1]
negatives = df[df["label"] == 0]
i = np.random.choice(negatives.index, len(negatives.index) - len(positives.index), replace=False)
df = df.drop(i)

write_df(df, root)