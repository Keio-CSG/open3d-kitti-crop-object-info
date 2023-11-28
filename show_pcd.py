import open3d as o3d
import numpy as np
import matplotlib.cm as cm

with open("data/KITTI/training/velodyne/000500.bin", "rb") as f:
    data = np.frombuffer(f.read(), dtype=np.float32).reshape(-1, 4)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data[:, :3])

cmap = cm.get_cmap("jet")
pcd.colors = o3d.utility.Vector3dVector(cmap(data[:, 3])[:, :3])
o3d.visualization.draw_geometries([pcd])
