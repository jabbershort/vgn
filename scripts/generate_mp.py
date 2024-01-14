from time import sleep
from multiprocessing import Queue, Pool, Process, current_process, cpu_count
from tqdm import tqdm
from vgn.grasp import Grasp, Label
from vgn.io import *
from vgn.perception import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform

OBJECT_COUNT_LAMBDA = 4
MAX_VIEWPOINT_COUNT = 6
GRASPS_PER_SCENE = 120

class DataGenerator:
    def __init__(self,root,scene="pile",object_set="blocks",count=10000):
        self.root = root
        self.grasp_count = count
        self.num_workers = cpu_count()
        self.sim = ClutterRemovalSim(scene, object_set)
        self.finger_depth = self.sim.gripper.finger_depth
        (self.root / "scenes").mkdir(parents=True, exist_ok=True)
        write_setup(
            self.root,
            self.sim.size,
            self.sim.camera.intrinsic,
            self.sim.gripper.max_opening_width,
            self.sim.gripper.finger_depth,
            )
        

    @property
    def scenes(self):
        return [*range(int(self.grasp_count/GRASPS_PER_SCENE)+1)]

    def worker_thread(self):
        while self.queue.qsize() >0:
            record = self.queue.get()
            print(f'Worker: {current_process().name}: {record}')
            self.generate_single()
            sleep(1)
        print(f'Worker: {current_process().name} finished')

    def run(self):
        pbar = tqdm(total=args.num_grasps)
        self.queue = Queue()
        for id in self.scenes:
            self.queue.put(id)
        self.processes = [Process(target=self.worker_thread) for _ in range(self.num_workers)]

        for process in self.processes:
            process.start()
            print('Process started')

        for process in self.processes:
            process.join()      

    def generate_single(self):
        # generate heap
        object_count = np.random.poisson(OBJECT_COUNT_LAMBDA) + 1
        self.sim.reset(object_count)
        self.sim.save_state()

        # render synthetic depth images
        n = np.random.randint(MAX_VIEWPOINT_COUNT) + 1
        depth_imgs, extrinsics = render_images(self.sim, n)

        # reconstrct point cloud using a subset of the images
        tsdf = create_tsdf(self.sim.size, 120, depth_imgs, self.sim.camera.intrinsic, extrinsics)
        pc = tsdf.get_cloud()

        # crop surface and borders from point cloud
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(self.sim.lower, self.sim.upper)
        pc = pc.crop(bounding_box)
        # o3d.visualization.draw_geometries([pc])

        if pc.is_empty():
            print("Point cloud empty, skipping scene")
            return

        # store the raw data
        scene_id = write_sensor_data(self.root, depth_imgs, extrinsics)

        for _ in range(GRASPS_PER_SCENE):
            # sample and evaluate a grasp point
            point, normal = sample_grasp_point(pc, self.finger_depth)
            grasp, label = evaluate_grasp_point(self.sim, point, normal)

            # store the sample
            write_grasp(self.root, scene_id, grasp, label)


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
    ok = False
    while not ok:
        # TODO this could result in an infinite loop, though very unlikely
        idx = np.random.randint(len(points))
        point, normal = points[idx], normals[idx]
        ok = normal[2] > -0.1  # make sure the normal is poitning upwards
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


if __name__ == '__main__':

    gen = DataGenerator('data/raw/test',1000)
    gen.run()
