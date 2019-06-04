import numpy as np

def depth_map_radial_to_planar(depth, K):
    """
    Transform depth maps from radial depth (euclidean distance to the camera center)
    to planar depth (orthogonal distance to the image plane + 1)
    """
    H = depth.shape[0]
    W = depth.shape[1]

    y = np.arange(0, H) + 0.5
    x = np.arange(0, W) + 0.5
    xv, yv = np.meshgrid(x, y)

    xs = xv.reshape(1,-1)
    ys = yv.reshape(1,-1)
    zs = np.ones(xs.shape)
    ds = depth.reshape(-1)

    locs_3d = np.concatenate((xs,ys,zs), axis=0)
    locs_3d = np.matmul(np.linalg.inv(K), locs_3d)
    nrms = np.linalg.norm(locs_3d, axis=0).reshape(1,-1)
    locs_3d = locs_3d / nrms
    locs_3d = locs_3d * ds

    zs = locs_3d[2]
    planar_depth = zs.reshape(depth.shape)
    return planar_depth

def depth_map_to_point_cloud(color_image, depth_map, camera, BGR=True, crop_fraction=0.0, vmin=None, vmax=None, add_alpha=False):
    """
    Project depth map pixels into space. Zero-depth pixels are ignored.

    color_image -- numpy array of the image, 3xHxW (0-1)
    depth_map -- numpy array of the image plane, 2D
    camera -- projection matrix for these pixels

    point_cloud -- (N,) numpy array with dtype (f4, f4, f4, u1, u1, u1)
    """
    H, W = depth_map.shape[:2]
    xs = np.linspace(0,W-1,W) + 0.5
    ys = np.linspace(0,H-1,H) + 0.5
    xs, ys = np.meshgrid(xs, ys)
    xs = xs.flatten()
    ys = ys.flatten()
    if BGR:
        rs = color_image[2].flatten()*255.0
        gs = color_image[1].flatten()*255.0
        bs = color_image[0].flatten()*255.0
    else:
        rs = color_image[0].flatten()*255.0
        gs = color_image[1].flatten()*255.0
        bs = color_image[2].flatten()*255.0
    if crop_fraction > 0:
        heightcrop = int(H*crop_fraction)
        widthcrop = int(W*crop_fraction)
        depth_map[:heightcrop,:] = 0
        depth_map[-heightcrop:,:] = 0
        depth_map[:,:widthcrop] = 0
        depth_map[:,-widthcrop:] = 0
    if vmin is not None:
        depth_map[depth_map < vmin] = 0
    if vmax is not None:
        depth_map[depth_map > vmax] = 0

    zs = depth_map.flatten()
    N = (zs > 0).sum()
    us = xs[zs > 0].reshape((1,N))
    vs = ys[zs > 0].reshape((1,N))
    rs = rs[zs > 0].reshape((1,N))
    gs = gs[zs > 0].reshape((1,N))
    bs = bs[zs > 0].reshape((1,N))
    os = np.ones((1,N))
    ps = np.concatenate((us,vs,os), axis=0)
    zs = zs[zs > 0].reshape((1,N))
    KR = camera[:3,:3]
    Kt = camera[:3,3:]
    cs = np.matmul(np.linalg.inv(KR), ps*zs - Kt)
    if not add_alpha:
        point_cloud = np.empty((N,),dtype=[
            ('x', 'float32'),('y', 'float32'), ('z', 'float32'),
            ('r', 'uint8'),('g', 'uint8'), ('b', 'uint8'),
        ])
        point_cloud['x'] = cs[0,:]
        point_cloud['y'] = cs[1,:]
        point_cloud['z'] = cs[2,:]
        point_cloud['r'] = rs.astype("uint8")
        point_cloud['g'] = gs.astype("uint8")
        point_cloud['b'] = bs.astype("uint8")
    else:
        point_cloud = np.empty((N,),dtype=[
            ('x', 'float32'),('y', 'float32'), ('z', 'float32'),
            ('r', 'uint8'),('g', 'uint8'), ('b', 'uint8'), ('a', 'uint8'),
        ])
        point_cloud['x'] = cs[0,:]
        point_cloud['y'] = cs[1,:]
        point_cloud['z'] = cs[2,:]
        point_cloud['r'] = rs.astype("uint8")
        point_cloud['g'] = gs.astype("uint8")
        point_cloud['b'] = bs.astype("uint8")
        point_cloud['a'] = 255

    return point_cloud