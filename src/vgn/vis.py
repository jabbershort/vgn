"""Render volumes, point clouds, and grasp detections in rviz."""

import matplotlib.colors
import numpy as np

from vgn.utils import workspace_lines
from vgn.utils.transform import Transform, Rotation
from vgn.grasp import Grasp
import open3d as o3d

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("RedGreen", ["r", "g"])

def draw_scene(size,cloud=None,tsdf=None,grasps=None,finger_depth=40/6.0, voxel_size=1, threshold=0.01,grid_origin=[0,0,0]):
    translation_array = [grid_origin[0],grid_origin[2],grid_origin[1]] # wtf
    geom = []
    if tsdf is not None:
        grid = generate_tsdf(tsdf,voxel_size,threshold)
        grid.translate(translation_array) # wtf
        geom.append(grid)
    if cloud is not None:   
        if type(cloud) == o3d.cpu.pybind.geometry.TriangleMesh:
            cloud.compute_vertex_normals()
        else:
            cloud.estimate_normals()
        geom.append(cloud)
    if grasps is not None:
        if type(grasps) == Grasp:
            grasp = generate_grasp(grasps,voxel_size)
            grasp.translate(translation_array) # wtf
            geom.append(grasp)
        if type(grasps) == list and len(grasps) != 0:
            for grasp in grasps:
                grasp = generate_grasp(grasp,voxel_size)
                grasp.translate(translation_array) # wtf
                geom.append(grasp)
    o3d.visualization.draw_geometries(geom)
    
def generate_tsdf(vol, voxel_size, threshold=0.01):
    vol = vol.squeeze()
    points = np.argwhere(vol > threshold) * voxel_size
    # values = np.expand_dims(vol[vol > threshold], 1)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return cloud

def draw_points(points):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    o3d.visualization.draw_geometries([cloud])

def generate_grasp(grasp,voxel_size,coordinate_size=10,scale=True):
    p = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coordinate_size)
    p.transform(grasp.pose.as_matrix())
    if scale:
        p.scale(voxel_size,center=[0,0,0])
    return p

