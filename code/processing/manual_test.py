import torch
import numpy as np
import cv2
import os
import glob
from utils.colmap_utils import read_cameras_binary, read_images_binary, qvec2rotmat
from utils.depth_maps import depth_map_to_point_cloud
from utils.ply import save_pointcloud_ply
from utils.depth_map_visualization import color_depth_map, color_trust_image
from experiments.continue_experiment import get_latest_instance_base
from utils.chamfer_evaluation import filter_cloud

from local_config import base_evaluation_output_path
from local_config import base_data_folder

subfolder = "ddf_qualitative_real/gnome_workspace/"
folder_in = os.path.join(base_data_folder, subfolder)
folder_out = os.path.join(base_evaluation_output_path, subfolder)
os.makedirs(folder_out, exist_ok=True)

vis_views = [6,]
save_clouds = True

images = {}
estimates = {}
cameras = {}
geo_estimates = {}

colmap_cameras = read_cameras_binary(os.path.join(
    folder_in,
    "sparse",
    "cameras.bin"
))
colmap_images = read_images_binary(os.path.join(
    folder_in,
    "sparse",
    "images.bin"
))

for image_idx in colmap_images:
    base_filename = colmap_images[image_idx].name
    image_filename = os.path.join(folder_in, "images", base_filename)
    images[base_filename] = cv2.imread(image_filename).astype(np.float32)/255.0
    estimates[base_filename] = np.load(os.path.join(
        folder_in,
        "stereo",
        "depth_maps",
        "photometric",
        base_filename + ".photometric.bin.npy"
    )).astype(np.float32)
    geo_estimates[base_filename] = np.load(os.path.join(
        folder_in,
        "stereo",
        "depth_maps",
        "geometric",
        base_filename + ".geometric.bin.npy"
    )).astype(np.float32)
    cam_idx = colmap_images[image_idx].camera_id
    cam = colmap_cameras[cam_idx]
    K = np.array([
        [cam.params[0],0,cam.params[2]],
        [0,cam.params[1],cam.params[3]],
        [0,0,1]
    ])
    R = qvec2rotmat(colmap_images[image_idx].qvec)
    t = colmap_images[image_idx].tvec.reshape(3,1)
    Rt = np.hstack((R, t))
    cameras[base_filename] = np.matmul(K, Rt).astype(np.float32)

for image_idx in colmap_images:
    base_filename = colmap_images[image_idx].name
    image_size = images[base_filename].shape
    resx = (image_size[1] // 32) * 32
    resy = (image_size[0] // 32) * 32
    images[base_filename] = images[base_filename][:resy, :resx]
    estimates[base_filename] = estimates[base_filename][:resy, :resx]
    estimates[base_filename] = estimates[base_filename][:resy, :resx]
    geo_estimates[base_filename] = geo_estimates[base_filename][:resy, :resx]

print("Finished loading all the data")

experiments = {
    'colmap_based': [
        'DTU_colmap_depth_refine_run1',
        'DTU_colmap_depth_refine_run2',
        'DTU_colmap_depth_refine_run3',
    ],
}

from experiment_handler import ExperimentHandler

images_list = []
estimates_list = []
geo_estimates_list = []
cameras_list = []

for image in images:
    images_list.append(torch.from_numpy(images[image].transpose((2,0,1))[None][None]).cuda())
    estimates_list.append(torch.from_numpy(estimates[image][None][None][None]).cuda())
    geo_estimates_list.append(torch.from_numpy(geo_estimates[image][None][None][None]).cuda())
    cameras_list.append(torch.from_numpy(cameras[image][None][None]).cuda())

images0 = images_list
estimates0 = estimates_list
geo_estimates0 = geo_estimates_list
cameras0 = cameras_list

views = list(range(len(images)))
print("Processing in differing scales to get a rough idea of where the network works best")
depth_scales = [0.3, 1.0, 6.0]

for depth_scale in depth_scales:
    print("Scaling world by a factor %f" % depth_scale)
    for experiment in experiments:
        run_results = [
            [images0, geo_estimates0, geo_estimates0, cameras0],
        ]
        runs = experiments[experiment]
        for run_idx, run in enumerate(runs):
            run = get_latest_instance_base(run)
            images = run_results[run_idx][0]
            estimates = run_results[run_idx][1]
            trusts = run_results[run_idx][2]
            cameras = run_results[run_idx][3]
            results = [images,[],[],cameras]
            with torch.no_grad():
                handler = ExperimentHandler.load_experiment_from_file(run)
                handler.network.depth_scale = depth_scale
                handler.network.scale_augmentation = 1.0

                for image_idx in views:
                    order = list(range(len(images)))
                    order[0], order[image_idx] = order[image_idx], order[0]
                    oimages = [images[idx] for idx in order]
                    oestimates = [estimates[idx] for idx in order]
                    otrusts = [trusts[idx] for idx in order]
                    ocameras = [cameras[idx] for idx in order]
                    image_tensor = torch.cat(oimages, dim=1)
                    estimate_tensor = torch.cat(oestimates, dim=1)
                    trust_tensor = torch.cat(otrusts, dim=1)
                    camera_tensor = torch.cat(ocameras, dim=1)
                    if run_idx == 0:
                        refined_estimate, refined_trust = handler.network(image_tensor, estimate_tensor, camera_tensor, do_trust=False)
                    else:
                        refined_estimate, refined_trust = handler.network(image_tensor, estimate_tensor, trust_tensor, camera_tensor)
                    results[1].append(refined_estimate[None])
                    results[2].append(refined_trust[None])
            
            run_results.append(results)
        
        orig_images = run_results[0][0]
        orig_colmap_photo = run_results[0][1]
        orig_colmap_geo = run_results[0][2]
        orig_cameras = run_results[0][3]

        for view in vis_views:
            image = orig_images[view][0,0].cpu().numpy()
            cv2.imwrite(os.path.join(
                str(folder_out),
                "view%d_input_estimate.png" % view
            ), color_depth_map(orig_colmap_photo[view][0,0,0].cpu().numpy(), scale=25))
            cloud = depth_map_to_point_cloud(image, orig_colmap_geo[view][0,0,0].cpu().numpy(), orig_cameras[view][0,0].cpu().numpy())
            save_pointcloud_ply(
                os.path.join(
                    str(folder_out),
                    "view%d_input_cloud.ply" % view
                ),
                cloud
            )

            for run_idx, run in enumerate(runs):
                refined_estimate = run_results[run_idx+1][1][view][0]
                refined_trust = run_results[run_idx+1][2][view][0]
                camera = run_results[run_idx+1][3][view]

                cv2.imwrite(os.path.join(
                    str(folder_out),
                    "view%d_%s_estimate_scale_%05d_run%d.png" % (view, experiment, (depth_scale * 100), run_idx)
                ), color_depth_map(refined_estimate[0,0].cpu().detach().numpy(), scale=25))
                cv2.imwrite(os.path.join(
                    str(folder_out),
                    "view%d_%s_trust_scale_%05d_run%d.png" % (view, experiment, (depth_scale * 100), run_idx)
                ), color_trust_image(refined_trust[0,0].cpu().detach().numpy()))
                trusted_depth = (refined_estimate * (refined_trust > 0.5).float()).detach().cpu().numpy()[0,0]
                cloud = depth_map_to_point_cloud(image, trusted_depth, camera[0,0].cpu().numpy())
                save_pointcloud_ply(
                    os.path.join(
                        str(folder_out),
                        "view%d_%s_cloud_scale_%05d_run%d.ply" % (view, experiment, (depth_scale * 100), run_idx)
                    ),
                    cloud
                )
        
        if save_clouds:
            cloud_evaluation_threshold = 0.01
            for trust_threshold in [0.75]:
                for run_idx, run in enumerate(runs):
                    cloud_locs = []
                    cloud_cols = []
                    for view in views:
                        input_image = run_results[run_idx+1][0][view][0,0]
                        refined_estimate = run_results[run_idx+1][1][view][0,0,0]
                        refined_trust = run_results[run_idx+1][2][view][0,0,0]
                        camera = run_results[run_idx+1][3][view][0,0]

                        trusted_depth = refined_estimate * (refined_trust > trust_threshold).float()
                        cloud = depth_map_to_point_cloud(input_image.cpu().numpy(), trusted_depth.cpu().numpy(), camera.cpu().numpy()).reshape(1,-1)
                        cloud_locs.append(
                            np.concatenate((
                                cloud['x'],
                                cloud['y'],
                                cloud['z'],
                            ),axis=0)
                        )
                        cloud_cols.append(
                            np.concatenate((
                                cloud['r'],
                                cloud['g'],
                                cloud['b'],
                            ),axis=0)
                        )
                    
                    cloud_locs = np.concatenate(cloud_locs, axis=1)
                    cloud_cols = np.concatenate(cloud_cols, axis=1)
                    cloud_locs, cloud_cols = filter_cloud(
                        cloud_locs, cloud_cols,
                        actual_threshold=cloud_evaluation_threshold/depth_scale,
                        actual_sparsity=cloud_evaluation_threshold/5/depth_scale
                    )

                    cloud = np.empty((cloud_locs.shape[1],),dtype=[
                        ('x', 'float32'),('y', 'float32'), ('z', 'float32'),
                        ('r', 'uint8'),('g', 'uint8'), ('b', 'uint8'),
                    ])
                    cloud['x'] = cloud_locs[0,:]
                    cloud['y'] = cloud_locs[1,:]
                    cloud['z'] = cloud_locs[2,:]
                    cloud['r'] = cloud_cols[0,:].astype("uint8")
                    cloud['g'] = cloud_cols[1,:].astype("uint8")
                    cloud['b'] = cloud_cols[2,:].astype("uint8")
                    save_pointcloud_ply(
                        os.path.join(
                            str(folder_out),
                            "full_cloud_%s_%05d_run%d.ply" % (experiment, depth_scale * 100, run_idx),
                        ),
                        cloud
                    )

print("Finished")