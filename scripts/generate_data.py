import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import scipy.signal as signal
from tqdm import tqdm

from vgn.grasp import Grasp, Label
from vgn.io import *
from vgn.perception import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform
from vgn.vis import generate_grasp
from vgn.parameters import *

def main(args):
    sim = ClutterRemovalSim(args.scene, args.object_set, gui=args.sim_gui)
    finger_depth = sim.gripper.finger_depth
    pbar = tqdm(total=args.num_grasps)

    if args.sim_gui:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

    (args.root / "scenes").mkdir(parents=True, exist_ok=True)
    write_setup(
        args.root,
        sim.size,
        sim.camera.intrinsic,
        sim.gripper.max_opening_width,
        sim.gripper.finger_depth,
    )

    for _ in range(args.num_grasps // DataGenerationParameters.grasps_per_scene):
        # generate heap
        object_count = np.random.poisson(DataGenerationParameters.num_objects) + 1
        sim.reset(object_count)
        sim.save_state()

        # render synthetic depth images
        n = np.random.randint(DataGenerationParameters.max_viewports) + 1
        depth_imgs, extrinsics = render_images(sim, n)

        # reconstrct point cloud using a subset of the images
        tsdf = create_tsdf(sim.size, 120, depth_imgs, sim.camera.intrinsic, extrinsics)
        pc = tsdf.get_cloud()

        # crop surface and borders from point cloud
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
        pc = pc.crop(bounding_box)
        # o3d.visualization.draw_geometries([pc])
        if args.sim_gui:
            vis.add_geometry(pc)

        if pc.is_empty():
            print("Point cloud empty, skipping scene")
            continue

        # store the raw data
        scene_id = write_sensor_data(args.root, depth_imgs, extrinsics)
        if args.sim_gui:
            vis.poll_events()
            vis.update_renderer()

        visible_grasps = []

        for _ in range(DataGenerationParameters.grasps_per_scene):
            # sample and evaluate a grasp point
            point, normal = sample_grasp_point(pc, finger_depth)
            grasp, label = evaluate_grasp_point(sim, point, normal)
            if label > 0:
                # TODO: parameterise numbers
                g = generate_grasp(grasp,label, 40.0 / 6.0,0.1)
                visible_grasps.append(g)
                if args.sim_gui:
                    vis.add_geometry(g)
                    vis.poll_events()
                    vis.update_renderer()
            # store the sample
            write_grasp(args.root, scene_id, grasp, label)
            pbar.update()

        if args.sim_gui:
            for g in visible_grasps:
                vis.remove_geometry(g)
        if args.sim_gui:
            vis.remove_geometry(pc,True)
    if args.sim_gui:
        vis.destroy_window()

    pbar.close()


def render_images(sim, n):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0])

    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)

    for i in range(n):
        r = np.random.uniform(1.6, 2.4) * sim.size
        theta = np.random.uniform(0.0, np.pi / 4.0)
        phi = np.random.uniform(0.0, 2.0 * np.pi)

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        depth_img = sim.camera.render(extrinsic)[1]

        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img

    return depth_imgs, extrinsics


def sample_grasp_point(point_cloud, finger_depth, eps=0.1):
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    angle_threshold = DataGenerationParameters.normal_threshold
    ok = False
    while not ok:
        # TODO this could result in an infinite loop, though very unlikely
        idx = np.random.randint(len(points))
        point, normal = points[idx], normals[idx]
        ok = np.abs(normal[0]) < angle_threshold and np.abs(normal[1]) < angle_threshold # make sure the normal is poitning upwards
    grasp_depth = np.random.uniform(-eps * finger_depth, (1.0 + eps) * finger_depth)
    point = point + normal * grasp_depth
    return point, normal


def evaluate_grasp_point(sim, pos, normal, num_rotations=6):
    # define initial grasp frame on object surface
    z_axis = -normal
    x_axis = np.r_[1.0, 0.0, 0.0]
    if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):
        x_axis = np.r_[0.0, 1.0, 0.0]
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = Rotation.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)

    # try to grasp with different yaw angles
    yaws = np.linspace(0.0, np.pi, num_rotations)
    outcomes, widths = [], []
    for yaw in yaws:
        ori = R * Rotation.from_euler("z", yaw)
        sim.restore_state()
        candidate = Grasp(Transform(ori, pos), width=sim.gripper.max_opening_width)
        outcome, width = sim.execute_grasp(candidate, remove=False)
        # if outcome == Label.SUCCESS:
        #     print("Success")
        outcomes.append(outcome)
        widths.append(width)

    # detect mid-point of widest peak of successful yaw angles
    # TODO currently this does not properly handle periodicity
    successes = (np.asarray(outcomes) == Label.SUCCESS).astype(float)
    if np.sum(successes):
        peaks, properties = signal.find_peaks(
            x=np.r_[0, successes, 0], height=1, width=1
        )
        idx_of_widest_peak = peaks[np.argmax(properties["widths"])] - 1
        ori = R * Rotation.from_euler("z", yaws[idx_of_widest_peak])
        width = widths[idx_of_widest_peak]

    return Grasp(Transform(ori, pos), width), int(np.max(outcomes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="pile")
    parser.add_argument("--object-set", type=str, default="blocks")
    parser.add_argument("--num-grasps", type=int, default=10000)
    parser.add_argument("--sim-gui", action="store_true")
    args = parser.parse_args()
    main(args)
