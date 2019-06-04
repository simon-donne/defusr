"""
Saving (and maybe one day loading) of .ply files
"""

import numpy as np

def save_pointcloud_ply(filename, point_cloud, alpha=False):
    """
    Save a point cloud to a ply file.
    Point cloud: np.array with dtype:
    {'names': ('x', 'y', 'z', 'r', 'g', 'b'), 'formats': ('f4', 'f4', 'f4', 'u1', 'u1', 'u1')}
    """
    with open(filename, "wt") as fh:
        fh.write("ply\n")
        fh.write("format binary_little_endian 1.0\n")
        fh.write("element vertex %d\n" % point_cloud.shape[0])
        fh.write("property float x\n")
        fh.write("property float y\n")
        fh.write("property float z\n")
        fh.write("property uchar red\n")
        fh.write("property uchar green\n")
        fh.write("property uchar blue\n")
        if alpha:
            fh.write("property uchar alpha\n")
        fh.write("end_header\n")
        point_cloud.tofile(fh)

def load_pointcloud_ply(filename):
    """
    Load a point from a ply file.
    Not flexible, only guaranteed to work on ply files from save_pointcloud_ply.
    Nobody said this was quick.
    """
    with open(filename, "rb") as fh:
        assert(fh.readline().rstrip() == b"ply")
        assert(fh.readline().rstrip() == b"format binary_little_endian 1.0")
        nr_elements = int(fh.readline().strip().decode('ascii').split(' ')[-1])
        assert(fh.readline().rstrip() == b"property float x")
        assert(fh.readline().rstrip() == b"property float y")
        assert(fh.readline().rstrip() == b"property float z")
        assert(fh.readline().rstrip() == b"property uchar red")
        assert(fh.readline().rstrip() == b"property uchar green")
        assert(fh.readline().rstrip() == b"property uchar blue")
        assert(fh.readline().rstrip() == b"end_header")
        data = np.fromfile(
            fh,
            dtype={'names': ('x', 'y', 'z', 'r', 'g', 'b'), 'formats': ('f4', 'f4', 'f4', 'u1', 'u1', 'u1')},
            count=6*nr_elements
        )
        data = np.rec.array(data, dtype=[('x', 'f4'),('y', 'f4'), ('z', 'f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
        data = data.reshape(1,-1)
        locs = np.concatenate((
            data['x'],
            data['y'],
            data['z'],
        ),axis=0)
        cols = np.concatenate((
            data['r'],
            data['g'],
            data['b'],
        ),axis=0)
        return locs, cols

def load_pointcloud_ply_meshlab(filename):
    """
    Load a point from a ply file.
    Nobody said this was quick.
    """
    with open(filename, "rb") as fh:
        assert(fh.readline().rstrip() == b"ply")
        assert(fh.readline().rstrip() == b"format binary_little_endian 1.0")
        fh.readline()
        nr_elements = int(fh.readline().strip().decode('ascii').split(' ')[-1])
        assert(fh.readline().rstrip() == b"property float x")
        assert(fh.readline().rstrip() == b"property float y")
        assert(fh.readline().rstrip() == b"property float z")
        assert(fh.readline().rstrip() == b"element face 0")
        assert(fh.readline().rstrip() == b"property list uchar int vertex_indices")
        assert(fh.readline().rstrip() == b"end_header")
        data = np.fromfile(
            fh,
            dtype={'names': ('x', 'y', 'z'), 'formats': ('f4', 'f4', 'f4')},
            count=6*nr_elements
        )
        data = np.rec.array(data, dtype=[('x', 'f4'),('y', 'f4'), ('z', 'f4')])
        data = data.reshape(1,-1)
        locs = np.concatenate((
            data['x'],
            data['y'],
            data['z'],
        ),axis=0)
        return locs

def load_surface_ply_dtu(filename):
    """
    Load a point from a ply file.
    Not flexible, only guaranteed to work on ply files from save_pointcloud_ply.
    Nobody said this was quick.
    """
    with open(filename, "rb") as fh:
        assert(fh.readline() == b"ply\n")
        assert(fh.readline() == b"format binary_little_endian 1.0\n")
        for _ in range(14):
            fh.readline()
        nr_elements = int(fh.readline().strip().decode('ascii').split(' ')[-1])
        assert(fh.readline() == b"property float x\n")
        assert(fh.readline() == b"property float y\n")
        assert(fh.readline() == b"property float z\n")
        assert(fh.readline() == b"property float value\n")
        nr_faces = int(fh.readline().strip().decode('ascii').split(' ')[-1])
        assert(fh.readline() == b"property list uchar int vertex_indices\n")
        assert(fh.readline() == b"end_header\n")
        data = np.fromfile(
            fh,
            dtype={'names': ('x', 'y', 'z', 'v'), 'formats': ('f4', 'f4', 'f4', 'f4')},
            count=nr_elements
        )
        data = np.rec.array(data, dtype=[('x', 'f4'),('y', 'f4'), ('z', 'f4'), ('v', 'f4')])
        data = data.reshape(1,-1)
        vertices = np.concatenate((
            data['x'],
            data['y'],
            data['z'],
        ),axis=0)

        data = np.fromfile(
            fh,
            dtype={'names': ('count', 'v1', 'v2', 'v3'), 'formats': ('u1', 'i4', 'i4', 'i4')},
            count=nr_faces
        )
        data = np.rec.array(data, dtype=[('count', 'u1'),('v1', 'i4'), ('v2', 'i4'), ('v3', 'i4')])
        data = data.reshape(1,-1)
        faces = np.concatenate((
            data['v1'],
            data['v2'],
            data['v3'],
        ),axis=0)

        return vertices, faces


def load_pointcloud_ply_gipuma(filename):
    """
    Load a point from a ply file, as written by gipuma (also stores normals).
    Not flexible.
    """
    with open(filename, "rb") as fh:
        assert(fh.readline().rstrip() == b"ply")
        assert(fh.readline().rstrip() == b"format binary_little_endian 1.0")
        nr_elements = int(fh.readline().strip().decode('ascii').split(' ')[-1])
        assert(fh.readline().rstrip() == b"property float x")
        assert(fh.readline().rstrip() == b"property float y")
        assert(fh.readline().rstrip() == b"property float z")
        assert(fh.readline().rstrip() == b"property float nx")
        assert(fh.readline().rstrip() == b"property float ny")
        assert(fh.readline().rstrip() == b"property float nz")
        assert(fh.readline().rstrip() == b"property uchar red")
        assert(fh.readline().rstrip() == b"property uchar green")
        assert(fh.readline().rstrip() == b"property uchar blue")
        assert(fh.readline().rstrip() == b"end_header")
        data = np.fromfile(
            fh,
            dtype={'names': ('x', 'y', 'z', 'nx', 'ny', 'nz', 'r', 'g', 'b'), 'formats': ('f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'u1', 'u1', 'u1')},
            count=6*nr_elements
        )
        data = np.rec.array(data, dtype=[('x', 'f4'),('y', 'f4'), ('z', 'f4'), ('nx', 'f4'),('ny', 'f4'), ('nz', 'f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
        data = data.reshape(1,-1)
        locs = np.concatenate((
            data['x'],
            data['y'],
            data['z'],
        ),axis=0)
        cols = np.concatenate((
            data['r'],
            data['g'],
            data['b'],
        ),axis=0)
        return locs, cols

def load_pointcloud_ply_ETH3D(filename):
    """
    Load a point from a ply file, as written by gipuma (also stores normals).
    Not flexible.
    """
    with open(filename, "rb") as fh:
        assert(fh.readline() == b"ply\n")
        assert(fh.readline() == b"format binary_little_endian 1.0\n")
        assert(fh.readline() == b"comment PCL generated\n")
        nr_elements = int(fh.readline().strip().decode('ascii').split(' ')[-1])
        assert(fh.readline() == b"property float x\n")
        assert(fh.readline() == b"property float y\n")
        assert(fh.readline() == b"property float z\n")
        assert(fh.readline() == b"element camera 1\n")
        for _ in range(21):
            fh.readline()
        assert(fh.readline() == b"end_header\n")
        data = np.fromfile(
            fh,
            dtype=np.float32,
            count=nr_elements * 3
        )
        data = data.reshape(-1, 3).transpose()
        return data
