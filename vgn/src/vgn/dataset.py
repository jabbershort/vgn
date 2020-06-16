import numpy as np
import pandas
from scipy import ndimage
import torch.utils.data

from vgn import from_voxel_coordinates
from vgn.grasp import Grasp
from vgn.utils.transform import Rotation, Transform
from vgn_ros import vis


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, augment=False):
        self.root = root
        self.df = pandas.read_csv(self.root / "grasps.csv")
        self._augment = augment

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, i):
        scene_id, index, rotation, width, label = self._lookup(i)
        tsdf = self._read_tsdf(scene_id)

        if self._augment:
            tsdf, index, rotation = self._apply_random_transform(tsdf, index, rotation)

        rotations = np.empty((2, 4), dtype=np.float32)
        R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
        rotations[0] = rotation.as_quat()
        rotations[1] = (rotation * R).as_quat()

        x, y, index = tsdf, (label, rotations, width), index

        return x, y, index

    def draw(self, i, finger_depth):
        size = 6.0 * finger_depth
        voxel_size = size / 40.0

        scene_id, index, rotation, width, label = self._lookup(i)
        tsdf = self._read_tsdf(scene_id)
        grasp = Grasp(Transform(rotation, index), width)
        grasp = from_voxel_coordinates(grasp, voxel_size)

        vis.clear()
        vis.workspace(size)
        vis.tsdf(tsdf.squeeze(), voxel_size)
        vis.grasps([grasp], [float(label)], finger_depth)

        cloud_path = self.root / "clouds" / (scene_id + ".npz")
        if cloud_path.exists():
            points = np.load(str(cloud_path))["points"]
            vis.points(points)

    def _lookup(self, i):
        scene_id = self.df.loc[i, "scene_id"]
        index = self.df.loc[i, "i":"k"].to_numpy(dtype=np.long)
        rotation = Rotation.from_quat(self.df.loc[i, "qx":"qw"].to_numpy())
        width = self.df.loc[i, "width"]
        label = self.df.loc[i, "label"]
        return scene_id, index, rotation, width, label

    def _read_tsdf(self, scene_id):
        tsdf_path = self.root / "tsdfs" / (scene_id + ".npz")
        return np.load(str(tsdf_path))["tsdf"]

    def _apply_random_transform(self, tsdf, index, rotation):
        # center sample at grasp point
        T_center = Transform(Rotation.identity(), index)
        # sample random transform
        angle = np.random.uniform(0.0, 2.0 * np.pi)
        R_augment = Rotation.from_rotvec(np.r_[0.0, 0.0, angle])
        t_augment = 20.0 - index + np.random.uniform(-16, 16, size=(3,))
        T_augment = Transform(R_augment, t_augment)
        T = T_center * T_augment * T_center.inverse()
        # transform tsdf
        T_inv = T.inverse()
        matrix, offset = T_inv.rotation.as_dcm(), T_inv.translation
        tsdf[0] = ndimage.affine_transform(tsdf[0], matrix, offset, order=1)
        # transform grasp pose
        index = np.round(T.transform_point(index)).astype(np.long)
        rotation = T.rotation * rotation
        return tsdf, index, rotation
