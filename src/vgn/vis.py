"""Render volumes, point clouds, and grasp detections in rviz."""

import matplotlib.colors
import numpy as np
# from sensor_msgs.msg import PointCloud2
# import rospy
# from rospy import Publisher
# from visualization_msgs.msg import Marker, MarkerArray

from vgn.utils import workspace_lines
from vgn.utils.transform import Transform, Rotation

import open3d as o3d

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("RedGreen", ["r", "g"])
# DELETE_MARKER_MSG = Marker(action=Marker.DELETEALL)
# DELETE_MARKER_ARRAY_MSG = MarkerArray(markers=[DELETE_MARKER_MSG])


def draw_workspace(size):
    scale = size * 0.005
    pose = Transform.identity()
    scale = [scale, 0.0, 0.0]
    color = [0.5, 0.5, 0.5]
    # msg = _create_marker_msg(Marker.LINE_LIST, "task", pose, scale, color)
    # msg.points = [ros_utils.to_point_msg(point) for point in workspace_lines(size)]
    # pubs["workspace"].publish(msg)


def draw_scene(size,cloud,tsdf,grasp,grasp_score,finger_depth, voxel_size=1, threshold=0.01):
    geom = []
    if tsdf is not None:
        grid = generate_tsdf(tsdf,voxel_size,threshold)
        geom.append(grid)
    if cloud is not None:   
        # tODO: parmaterise functions
        cloud = cloud.scale(40/size,[0,0,0])
        cloud.translate([0,0,2])
        cloud.compute_vertex_normals()
        geom.append(cloud)
    grip = generate_grasp(grasp,grasp_score,finger_depth)
    geom.append(grip)
    o3d.visualization.draw_geometries(geom)
    
def generate_tsdf(vol, voxel_size, threshold=0.01):
    vol = vol.squeeze()
    points = np.argwhere(vol > threshold) * voxel_size
    values = np.expand_dims(vol[vol > threshold], 1)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return cloud


def draw_points(points):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    o3d.visualization.draw_geometries([cloud])


# def draw_quality(vol, voxel_size, threshold=0.01):
    # msg = _create_vol_msg(vol, voxel_size, threshold)
    # pubs["quality"].publish(msg)


# def draw_volume(vol, voxel_size, threshold=0.01):
    # msg = _create_vol_msg(vol, voxel_size, threshold)
    # pubs["debug"].publish(msg)

def generate_grasp(grasp,score,finger_depth,coordinate_size=10):
    p = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coordinate_size)
    p.transform(grasp.pose.as_matrix())
    return p

def draw_grasp(grasp, score, finger_depth):
    radius = 0.1 * finger_depth
    w, d = grasp.width, finger_depth
    color = cmap(float(score))

    markers = []

    # left finger
    pose = grasp.pose * Transform(Rotation.identity(), [0.0, -w / 2, d / 2])
    scale = [radius, radius, d]
    # msg = _create_marker_msg(Marker.CYLINDER, "task", pose, scale, color)
    # msg.id = 0
    # markers.append(msg)

    # right finger
    pose = grasp.pose * Transform(Rotation.identity(), [0.0, w / 2, d / 2])
    scale = [radius, radius, d]
    # msg = _create_marker_msg(Marker.CYLINDER, "task", pose, scale, color)
    # msg.id = 1
    # markers.append(msg)

    # wrist
    pose = grasp.pose * Transform(Rotation.identity(), [0.0, 0.0, -d / 4])
    scale = [radius, radius, d / 2]
    # msg = _create_marker_msg(Marker.CYLINDER, "task", pose, scale, color)
    # msg.id = 2
    # markers.append(msg)

    # palm
    pose = grasp.pose * Transform(
        Rotation.from_rotvec(np.pi / 2 * np.r_[1.0, 0.0, 0.0]), [0.0, 0.0, 0.0]
    )
    scale = [radius, radius, w]
    # msg = _create_marker_msg(Marker.CYLINDER, "task", pose, scale, color)
    # msg.id = 3
    # markers.append(msg)

    # pubs["grasp"].publish(MarkerArray(markers=markers))



