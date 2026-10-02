"""Step 2: turn depth images into 3D points in room coordinates (one fused point cloud)."""
import numpy as np
import open3d as o3d


def backproject(depth, K, T_wc, max_depth: float = 4.0):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    m = (depth > 0.1) & (depth < max_depth)
    z = depth[m]
    x = (u[m] - K[0, 2]) / K[0, 0] * z
    y = (v[m] - K[1, 2]) / K[1, 1] * z
    cam = np.stack([x, y, z], axis=1)
    return cam @ T_wc[:3, :3].T + T_wc[:3, 3]


def fuse_points(cap, stride: int = 5, max_depth: float = 4.0, min_conf: int = 2, voxel: float = 0.02):
    chunks = [backproject(cap.load_depth(f.index, min_conf), f.K, f.T_wc, max_depth)
              for f in cap.frames[::stride]]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(chunks)))
    return np.asarray(pc.voxel_down_sample(voxel).points)


def estimate_normals(points, radius: float = 0.08, max_nn: int = 20):
    """Surface direction at each point, from its neighbours (sign is arbitrary)."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    return np.asarray(pc.normals)
