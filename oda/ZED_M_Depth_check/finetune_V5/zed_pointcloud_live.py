import pyzed.sl as sl
import numpy as np
import open3d as o3d

zed = sl.Camera()
init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.HD1080
init.depth_mode = sl.DEPTH_MODE.NEURAL
init.coordinate_units = sl.UNIT.METER
init.depth_minimum_distance = 0.2

status = zed.open(init)
print("open() status:", status)
if status != sl.ERROR_CODE.SUCCESS:
    exit(1)

runtime = sl.RuntimeParameters()
point_cloud = sl.Mat()

vis = o3d.visualization.Visualizer()
vis.create_window("ZED Mini - Live Neural Depth Point Cloud (close window to quit)", width=1280, height=800)

pcd = o3d.geometry.PointCloud()
first_frame = True

print("Streaming point cloud. Close the window to quit.")

try:
    while True:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        pc_np = point_cloud.get_data()

        xyz = pc_np[:, :, :3].reshape(-1, 3)
        rgba_packed = pc_np[:, :, 3].reshape(-1)

        valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0)
        xyz = xyz[valid]
        rgba_packed = rgba_packed[valid]

        rgba_bytes = rgba_packed.copy().view(np.uint8).reshape(-1, 4)
        colors = rgba_bytes[:, :3].astype(np.float64) / 255.0

        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors)

        if first_frame:
            vis.add_geometry(pcd)
            first_frame = False
        else:
            vis.update_geometry(pcd)

        if not vis.poll_events():
            break
        vis.update_renderer()
finally:
    vis.destroy_window()
    zed.close()
    print("Viewer closed, camera released.")
