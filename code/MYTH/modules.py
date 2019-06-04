"""
Module providing wrappers to the MYTH FFI calls, in the form of simple functions,
torch.nn.Modules, or whatever else is applicable.
"""

import pickle
import torch
import MYTH
import pickle
from utils.timer import Timer

class AverageReprojectionPooling(torch.autograd.Function):
    """
    Average pool the inputs. Zero-entries are ignored.
    Ignores the first input camera, as that is assumed to be the center camera.

    Arguments:
        bounds -- the depth lower bounds (B x N x 1 x H x W)
        depths -- the depth reprojections (B x N x 1 x H x W)
        features -- the feature reprojections (B x N x F x H x W)

    Outputs:
        avg_bounds -- the depth lower bounds (B x 1 x H x W)
        avg_depths -- the depth reprojections (B x 1 x H x W)
        avg_features -- the feature reprojections (B x F x H x W)
    """

    @staticmethod
    def forward(ctx, bounds, depths, features):
        B = features.shape[0]
        N = features.shape[1]
        F = features.shape[2]
        H = features.shape[3]
        W = features.shape[4]

        avg_bounds = bounds.new_zeros((B, 1, H, W))
        avg_depths = bounds.new_zeros((B, 1, H, W))
        avg_features = bounds.new_zeros((B, F, H, W))

        MYTH.AverageReprojectionPooling_updateOutput_gpu(
            bounds, depths, features,
            avg_bounds, avg_depths, avg_features
        )

        ctx.save_for_backward(bounds, depths)
        return avg_bounds, avg_depths, avg_features

    @staticmethod
    def backward(ctx, dloss_avg_bounds, dloss_avg_depths, dloss_avg_features):
        bounds, depths = ctx.saved_variables
        B = dloss_avg_features.shape[0]
        F = dloss_avg_features.shape[1]
        H = dloss_avg_features.shape[2]
        W = dloss_avg_features.shape[3]
        N = bounds.shape[1]

        dloss_bounds = dloss_avg_bounds.new_zeros((B, N, 1, H, W))
        dloss_depths = dloss_avg_bounds.new_zeros((B, N, 1, H, W))
        dloss_features = dloss_avg_bounds.new_zeros((B, N, F, H, W))

        MYTH.AverageReprojectionPooling_updateGradInput_gpu(
            bounds, depths,
            dloss_bounds, dloss_depths, dloss_features,
            dloss_avg_bounds, dloss_avg_depths, dloss_avg_features
        )

        return dloss_bounds, dloss_depths, dloss_features


class VariationReprojectionPooling(torch.autograd.Function):
    """
    Maximum pool the inputs, with respect to a culling map.
    Ignores the first input camera, as that is assumed to be the center camera.

    Arguments:
        bounds -- the depth lower bounds (B x N x 1 x H x W)
        depths -- the depth reprojections (B x N x 1 x H x W)
        features -- the feature reprojections (B x N x F x H x W)
        cull_mask -- the validity mask (B x N x 1 x H x W)

    Outputs:
        max_bounds -- the depth lower bounds (B x 1 x H x W)
        max_depths -- the depth reprojections (B x 1 x H x W)
        max_features -- the feature reprojections (B x F x H x W)
    """

    @staticmethod
    def forward(ctx, bounds, depths, avg_bounds, avg_depths, cull_mask):
        B = bounds.shape[0]
        N = bounds.shape[1]
        H = bounds.shape[3]
        W = bounds.shape[4]

        var_bounds = bounds.new_zeros((B, 1, H, W))
        var_depths = bounds.new_zeros((B, 1, H, W))

        MYTH.VariationReprojectionPooling_updateOutput_gpu(
            avg_bounds, avg_depths,
            bounds, depths,
            var_bounds, var_depths,
            cull_mask
        )

        ctx.save_for_backward(bounds, depths, avg_bounds, avg_depths, cull_mask)
        return var_bounds, var_depths

    @staticmethod
    def backward(ctx, dloss_var_bounds, dloss_var_depths):
        bounds, depths, avg_bounds, avg_depths, cull_mask = ctx.saved_variables
        B = dloss_var_bounds.shape[0]
        F = dloss_var_bounds.shape[1]
        H = dloss_var_bounds.shape[2]
        W = dloss_var_bounds.shape[3]
        N = cull_mask.shape[1]

        dloss_bounds = dloss_var_bounds.new_zeros((B, N, 1, H, W))
        dloss_depths = dloss_var_bounds.new_zeros((B, N, 1, H, W))
        dloss_avg_bounds = dloss_var_bounds.new_zeros((B, 1, H, W))
        dloss_avg_depths = dloss_var_bounds.new_zeros((B, 1, H, W))

        MYTH.VariationReprojectionPooling_updateGradInput_gpu(
            avg_bounds, avg_depths,
            bounds, depths,
            dloss_bounds, dloss_depths,
            dloss_avg_bounds, dloss_avg_depths,
            dloss_var_bounds, dloss_var_depths,
            cull_mask
        )

        return dloss_bounds, dloss_depths, dloss_avg_bounds, dloss_avg_depths, None


class MaximumReprojectionPooling(torch.autograd.Function):
    """
    Maximum pool the inputs, with respect to a culling map.
    Ignores the first input camera, as that is assumed to be the center camera.

    Arguments:
        bounds -- the depth lower bounds (B x N x 1 x H x W)
        depths -- the depth reprojections (B x N x 1 x H x W)
        features -- the feature reprojections (B x N x F x H x W)
        cull_mask -- the validity mask (B x N x 1 x H x W)

    Outputs:
        max_bounds -- the depth lower bounds (B x 1 x H x W)
        max_depths -- the depth reprojections (B x 1 x H x W)
        max_features -- the feature reprojections (B x F x H x W)
    """

    @staticmethod
    def forward(ctx, bounds, depths, features, cull_mask):
        B = features.shape[0]
        N = features.shape[1]
        F = features.shape[2]
        H = features.shape[3]
        W = features.shape[4]

        max_bounds = bounds.new_zeros((B, 1, H, W))
        max_depths = bounds.new_zeros((B, 1, H, W))
        max_features = bounds.new_zeros((B, F, H, W))

        MYTH.MaximumReprojectionPooling_updateOutput_gpu(
            bounds, depths, features,
            max_bounds, max_depths, max_features,
            cull_mask
        )

        ctx.save_for_backward(bounds, depths, cull_mask)
        return max_bounds, max_depths, max_features

    @staticmethod
    def backward(ctx, dloss_max_bounds, dloss_max_depths, dloss_max_features):
        bounds, depths, cull_mask = ctx.saved_variables
        B = dloss_max_features.shape[0]
        F = dloss_max_features.shape[1]
        H = dloss_max_features.shape[2]
        W = dloss_max_features.shape[3]
        N = cull_mask.shape[1]

        dloss_bounds = dloss_max_bounds.new_zeros((B, N, 1, H, W))
        dloss_depths = dloss_max_bounds.new_zeros((B, N, 1, H, W))
        dloss_features = dloss_max_bounds.new_zeros((B, N, F, H, W))

        MYTH.MaximumReprojectionPooling_updateGradInput_gpu(
            bounds, depths,
            dloss_bounds, dloss_depths, dloss_features,
            dloss_max_bounds, dloss_max_depths, dloss_max_features,
            cull_mask
        )

        return dloss_bounds, dloss_depths, dloss_features, None


class MinimumAbsReprojectionPooling(torch.autograd.Function):
    """
    Minimum pool the inputs' magnitudes, but keep the sign intact (with respect to a culling map).
    Ignores the first input camera, as that is assumed to be the center camera.

    Arguments:
        bounds -- the depth lower bounds (B x N x 1 x H x W)
        depths -- the depth reprojections (B x N x 1 x H x W)
        features -- the feature reprojections (B x N x F x H x W)
        cull_mask -- the validity mask (B x N x 1 x H x W)

    Outputs:
        min_bounds -- the depth lower bounds (B x 1 x H x W)
        min_depths -- the depth reprojections (B x 1 x H x W)
        min_features -- the feature reprojections (B x F x H x W)
    """

    @staticmethod
    def forward(ctx, bounds, depths, features, cull_mask):
        B = features.shape[0]
        N = features.shape[1]
        F = features.shape[2]
        H = features.shape[3]
        W = features.shape[4]

        max_bounds = bounds.new_zeros((B, 1, H, W))
        max_depths = bounds.new_zeros((B, 1, H, W))
        max_features = bounds.new_zeros((B, F, H, W))

        MYTH.MinimumAbsReprojectionPooling_updateOutput_gpu(
            bounds, depths, features,
            max_bounds, max_depths, max_features,
            cull_mask
        )

        ctx.save_for_backward(bounds, depths, cull_mask)
        return max_bounds, max_depths, max_features

    @staticmethod
    def backward(ctx, dloss_max_bounds, dloss_max_depths, dloss_max_features):
        bounds, depths, cull_mask = ctx.saved_variables
        B = dloss_max_features.shape[0]
        F = dloss_max_features.shape[1]
        H = dloss_max_features.shape[2]
        W = dloss_max_features.shape[3]
        N = cull_mask.shape[1]

        dloss_bounds = dloss_max_bounds.new_zeros((B, N, 1, H, W))
        dloss_depths = dloss_max_bounds.new_zeros((B, N, 1, H, W))
        dloss_features = dloss_max_bounds.new_zeros((B, N, F, H, W))

        MYTH.MinimumAbsReprojectionPooling_updateGradInput_gpu(
            bounds, depths,
            dloss_bounds, dloss_depths, dloss_features,
            dloss_max_bounds, dloss_max_depths, dloss_max_features,
            cull_mask
        )

        return dloss_bounds, dloss_depths, dloss_features, None


class DepthReprojectionCompleteBound(torch.autograd.Function):
    """
    Complete depth reprojections with bounds

    The first camera is assumed to be the center camera.
    All of the other view's points have reprojected onto that image plane.
    This module fills in empty depth pixels in the reprojections with bounds (per-neighbour).

    Arguments:
        depths -- the input views (B x N x 1 x H x W)
        reprojected -- the reprojected depth images to be completed (B x N x 1 x H x W)
        cameras -- their cameras (B x N x 3 x 4)
        scale -- when the images have been scaled compared to the input camera matrices

    Outputs:
        reprojected -- the reprojected depths (B x N x 1 x H x W)
    """

    @staticmethod
    def forward(ctx, depths, reprojected, cameras, scale, dmin, dmax, dstep):
        B = depths.shape[0]
        N = depths.shape[1]
        H = depths.shape[3]
        W = depths.shape[4]

        camlocs = depths.new_empty(B, N, 3, 1)
        invKRs = depths.new_empty(B, N, 3, 3)
        if scale != 1.0:
            for b in range(B):
                for n in range(N):
                    cameras[b][n] = cameras[b][n].clone()
                    cameras[b][n][:2,:] = cameras[b][n][:2,:] * scale
                    invKRs[b][n] = torch.inverse(cameras[b][n][:3, :3]).contiguous()
                    camlocs[b][n] = - torch.mm(invKRs[b][n], cameras[b][n][:3, 3:4])
        else:
            MYTH.InvertCams_gpu(cameras.reshape(-1,3,4), invKRs.reshape(-1,3,3), camlocs.reshape(-1,3,1))

        ctx.save_for_backward(depths, cameras, camlocs, invKRs)
        MYTH.DepthReprojectionCompleteBound_updateOutput_gpu(depths, reprojected, cameras, invKRs, camlocs, dmin, dmax, dstep)

        return reprojected

    @staticmethod
    def backward(ctx, gradOutput):
        # note: backprop not implemented yet (if ever)
        return None, None, None, None, None, None, None

class DepthReprojectionNonzeroCompleteBound(torch.autograd.Function):
    """
    Complete depth reprojections with bounds

    The first camera is assumed to be the center camera.
    All of the other view's points have reprojected onto that image plane.
    This module fills in empty depth pixels in the reprojections with bounds (per-neighbour).
    As soon as it hits an empty pixel for a given neighbour, the bound ends there: that is unknown
    and we can't assume it empty.

    Arguments:
        depths -- the input views (B x N x 1 x H x W)
        reprojected -- the reprojected depth images to be completed (B x N x 1 x H x W)
        cameras -- their cameras (B x N x 3 x 4)
        scale -- when the images have been scaled compared to the input camera matrices

    Outputs:
        reprojected -- the reprojected depths (B x N x 1 x H x W)
    """

    @staticmethod
    def forward(ctx, depths, reprojected, cameras, scale, dmin, dmax, dstep):
        B = depths.shape[0]
        N = depths.shape[1]
        H = depths.shape[3]
        W = depths.shape[4]

        camlocs = depths.new_empty(B, N, 3, 1)
        invKRs = depths.new_empty(B, N, 3, 3)
        if scale != 1.0:
            for b in range(B):
                for n in range(N):
                    cameras[b][n] = cameras[b][n].clone()
                    cameras[b][n][:2,:] = cameras[b][n][:2,:] * scale
                    invKRs[b][n] = torch.inverse(cameras[b][n][:3, :3]).contiguous()
                    camlocs[b][n] = - torch.mm(invKRs[b][n], cameras[b][n][:3, 3:4])
        else:
            MYTH.InvertCams_gpu(cameras.reshape(-1,3,4), invKRs.reshape(-1,3,3), camlocs.reshape(-1,3,1))

        ctx.save_for_backward(depths, cameras, camlocs, invKRs)
        MYTH.DepthReprojectionNonzeroCompleteBound_updateOutput_gpu(depths, reprojected, cameras, invKRs, camlocs, dmin, dmax, dstep)

        return reprojected

    @staticmethod
    def backward(ctx, gradOutput):
        # note: backprop not implemented yet (if ever)
        return None, None, None, None, None, None, None


class DepthColorAngleReprojectionNeighbours(torch.autograd.Function):
    """
    Neighbour depth reprojection

    The first camera is assumed to be the center camera.
    All of the other view's points are reprojected onto that image plane.
    Zero-depth pixels are ignored, the first depth map is obviously left unchanged.

    Colour is also being reprojected along (following that zbuffer)

    Arguments:
        images -- the input views (B x N x C x H x W)
        depths -- the input views (B x N x 1 x H x W)
        cameras -- their cameras (B x N x 3 x 4)
        scale -- when the images have been scaled compared to the input camera matrices

    Outputs:
        output_color -- the reprojected colors (B x N x C x H x W)
        output_depth -- the reprojected depths (B x N x 1 x H x W)
    """

    @staticmethod
    def forward(ctx, depths, images, cameras, scale):
        B = depths.shape[0]
        N = depths.shape[1]
        C = images.shape[2]
        H = depths.shape[3]
        W = depths.shape[4]

        sentinel = 1e9
        output_depth = depths.new_full((B, N, 1, H, W), fill_value=sentinel)
        output_color = images.new_full((B, N, C, H, W), fill_value=0.0)
        output_angle = depths.new_full((B, N, 1, H, W), fill_value=0.0)

        camlocs = depths.new_empty(B, N, 3, 1)
        invKRs = depths.new_empty(B, N, 3, 3)
        if scale != 1.0:
            for b in range(B):
                for n in range(N):
                    cameras[b][n] = cameras[b][n].clone()
                    cameras[b][n][:2,:] = cameras[b][n][:2,:] * scale
                    invKRs[b][n] = torch.inverse(cameras[b][n][:3, :3]).contiguous()
                    camlocs[b][n] = - torch.mm(invKRs[b][n], cameras[b][n][:3, 3:4])
        else:
            MYTH.InvertCams_gpu(cameras.reshape(-1,3,4), invKRs.reshape(-1,3,3), camlocs.reshape(-1,3,1))

        MYTH.DepthColorAngleReprojectionNeighbours_updateOutput_gpu(depths, output_depth, images, output_color, output_angle, cameras, invKRs, camlocs)

        output_depth[ output_depth > sentinel / 10] = 0
        output_angle[output_angle != output_angle] = 0

        ctx.save_for_backward(depths, output_depth, cameras.clone(), camlocs.clone(), invKRs.clone())

        return output_depth, output_color, output_angle

    @staticmethod
    def backward(ctx, grad_output_depth, grad_output_color, grad_output_angle):
        # note: backprop currently only implemented for the color
        depths, output_depth, cameras, camlocs, invKRs = ctx.saved_variables

        grad_input_color = grad_output_color.new_zeros(grad_output_color.shape)

        MYTH.DepthColorAngleReprojectionNeighbours_updateGradInput_gpu(
            depths, output_depth, grad_input_color, grad_output_color, cameras, invKRs, camlocs
        )

        return None, grad_input_color, None, None


class DepthReprojectionNeighbours(torch.autograd.Function):
    """
    Neighbour depth reprojection

    The first camera is assumed to be the center camera.
    All of the other view's points are reprojected onto that image plane.
    Zero-depth pixels are ignored, the first depth map is obviously left unchanged.

    Arguments:
        depths -- the input views (B x N x 1 x H x W)
        cameras -- their cameras (B x N x 3 x 4)
        scale -- when the images have been scaled compared to the input camera matrices

    Outputs:
        output_depth -- the reprojected depths (B x N x 1 x H x W)
    """

    @staticmethod
    def forward(ctx, depths, cameras, scale):
        B = depths.shape[0]
        N = depths.shape[1]
        H = depths.shape[3]
        W = depths.shape[4]

        sentinel = 1e9
        output_depth = depths.new_full((B, N, 1, H, W), fill_value=sentinel)

        camlocs = depths.new_empty(B, N, 3, 1)
        invKRs = depths.new_empty(B, N, 3, 3)
        if scale != 1.0:
            for b in range(B):
                for n in range(N):
                    cameras[b][n] = cameras[b][n].clone()
                    cameras[b][n][:2,:] = cameras[b][n][:2,:] * scale
                    invKRs[b][n] = torch.inverse(cameras[b][n][:3, :3]).contiguous()
                    camlocs[b][n] = - torch.mm(invKRs[b][n], cameras[b][n][:3, 3:4])
        else:
            MYTH.InvertCams_gpu(cameras.reshape(-1,3,4), invKRs.reshape(-1,3,3), camlocs.reshape(-1,3,1))

        MYTH.DepthReprojectionNeighbours_updateOutput_gpu(depths, output_depth, cameras, invKRs, camlocs)

        output_depth[ output_depth > sentinel / 10] = 0

        return output_depth

    @staticmethod
    def backward(ctx, grad_output_depth):
        # not differentiable right now
        return None, None, None, None


class PatchedReprojectionNeighbours(torch.autograd.Function):
    """
    Neighbour depth reprojection

    Zero-depth pixels are ignored, the first depth map is obviously left unchanged.

    Colour is also being reprojected along (following that zbuffer)

    Arguments:
        depth_center -- the center depth (B x 1 x H x W)
        color_center -- the center color (B x C x H x W)
        (above two are only used for show, to get the sizes of the reprojection)
        camera_center -- the center camera (B x 3 x 4)
        depths_neighbours -- the neighbouring depth images (B x N x 1 x H x W)
        colors_neighbours -- the neighbouring color images (B x N x C x H x W)
        cameras_neighbours -- the neighbouring cameras (B x N x 3 x 4)

    Outputs:
        output_color -- the reprojected colors (B x N x C x H x W)
        output_depth -- the reprojected depths (B x N x 1 x H x W)
    """

    @staticmethod
    def forward(ctx, trusted_depth_center, color_center, camera_center, trusted_depths_neighbours, colors_neighbours, cameras_neighbours):
        B = colors_neighbours.shape[0]
        N = colors_neighbours.shape[1]
        C = colors_neighbours.shape[2]
        H_out = color_center.shape[2]
        W_out = color_center.shape[3]

        sentinel = 1e9
        output_depth = trusted_depth_center.new_full((B, N, 1, H_out, W_out), fill_value=sentinel)
        output_color = color_center.new_full((B, N, C, H_out, W_out), fill_value=0.0)

        invKRs_neighbours = cameras_neighbours.new_empty(B, N, 3, 3)
        camlocs_neighbours = cameras_neighbours.new_empty(B, N, 3, 1)
        MYTH.InvertCams_gpu(cameras_neighbours.reshape(-1,3,4), invKRs_neighbours.reshape(-1,3,3), camlocs_neighbours.reshape(-1,3,1))

        MYTH.PatchedReprojectionNeighbours_updateOutput_gpu(
            trusted_depths_neighbours,
            colors_neighbours,
            output_depth,
            output_color,
            camera_center,
            invKRs_neighbours,
            camlocs_neighbours
        )

        output_depth[ output_depth > sentinel / 10] = 0

        ctx.save_for_backward(trusted_depths_neighbours, output_depth, camera_center.clone(), invKRs_neighbours.clone(), camlocs_neighbours.clone(), colors_neighbours.clone())

        return output_depth, output_color

    @staticmethod
    def backward(ctx, dloss_doutput_depth, dloss_doutput_color):
        # note: backprop currently only implemented for the color
        trusted_depths_neighbours, output_depth, camera_center, invKRs_neighbours, camlocs_neighbours, colors_neighbours = ctx.saved_variables

        dloss_dinput_color = dloss_doutput_color.new_zeros(colors_neighbours.shape)

        MYTH.PatchedReprojectionNeighbours_updateGradInput_gpu(
            trusted_depths_neighbours,
            output_depth,
            dloss_dinput_color,
            dloss_doutput_color,
            camera_center,
            invKRs_neighbours,
            camlocs_neighbours
        )

        return None, None, None, None, dloss_dinput_color, None


class PatchedReprojectionCompleteBound(torch.autograd.Function):
    """
    Complete depth reprojections with bounds

    The first camera is assumed to be the center camera.
    All of the other view's points have reprojected onto that image plane.
    This module fills in empty depth pixels in the reprojections with bounds (per-neighbour).

    Arguments:
        depths -- the input views (B x N x 1 x H x W)
        reprojected -- the reprojected depth images to be completed (B x N x 1 x H x W)
        cameras -- their cameras (B x N x 3 x 4)
        scale -- when the images have been scaled compared to the input camera matrices

    Outputs:
        reprojected -- the reprojected depths (B x N x 1 x H x W)
    """

    @staticmethod
    def forward(ctx, depth_center, camera_center, depths_neighbours, cameras_neighbours, dmin, dmax, dstep):
        B = depths_neighbours.shape[0]
        N = depths_neighbours.shape[1]
        H_center = depth_center.shape[2]
        W_center = depth_center.shape[3]

        invKR_center = camera_center.new_empty(B, 3, 3)
        camloc_center = camera_center.new_empty(B, 3, 1)
        MYTH.InvertCams_gpu(camera_center, invKR_center, camloc_center)

        bounds_neighbours = depth_center.new_zeros((B, N, 1, H_center, W_center))

        MYTH.PatchedReprojectionCompleteBound_updateOutput_gpu(
            depth_center,
            depths_neighbours,
            bounds_neighbours,
            cameras_neighbours,
            invKR_center,
            camloc_center,
            dmin,
            dmax,
            dstep
        )

        return bounds_neighbours

    @staticmethod
    def backward(ctx, gradOutput):
        return None, None, None, None, None, None, None, None

###########
# The only difference with these pooling methods is that they include the first neighbour in the pooling!
###########

class PatchedAverageReprojectionPooling(torch.autograd.Function):
    """
    Average pool the inputs. Zero-entries are ignored.
    Ignores the first input camera, as that is assumed to be the center camera.

    Arguments:
        bounds -- the depth lower bounds (B x N x 1 x H x W)
        depths -- the depth reprojections (B x N x 1 x H x W)
        features -- the feature reprojections (B x N x F x H x W)

    Outputs:
        avg_bounds -- the depth lower bounds (B x 1 x H x W)
        avg_depths -- the depth reprojections (B x 1 x H x W)
        avg_features -- the feature reprojections (B x F x H x W)
    """

    @staticmethod
    def forward(ctx, bounds, depths, features):
        B = features.shape[0]
        N = features.shape[1]
        F = features.shape[2]
        H = features.shape[3]
        W = features.shape[4]

        avg_bounds = bounds.new_zeros((B, 1, H, W))
        avg_depths = bounds.new_zeros((B, 1, H, W))
        avg_features = bounds.new_zeros((B, F, H, W))

        MYTH.PatchedAverageReprojectionPooling_updateOutput_gpu(
            bounds, depths, features,
            avg_bounds, avg_depths, avg_features
        )

        ctx.save_for_backward(bounds, depths)
        return avg_bounds, avg_depths, avg_features

    @staticmethod
    def backward(ctx, dloss_avg_bounds, dloss_avg_depths, dloss_avg_features):
        bounds, depths = ctx.saved_variables
        B = dloss_avg_features.shape[0]
        F = dloss_avg_features.shape[1]
        H = dloss_avg_features.shape[2]
        W = dloss_avg_features.shape[3]
        N = bounds.shape[1]

        dloss_bounds = dloss_avg_bounds.new_zeros((B, N, 1, H, W))
        dloss_depths = dloss_avg_bounds.new_zeros((B, N, 1, H, W))
        dloss_features = dloss_avg_bounds.new_zeros((B, N, F, H, W))

        MYTH.PatchedAverageReprojectionPooling_updateGradInput_gpu(
            bounds, depths,
            dloss_bounds, dloss_depths, dloss_features,
            dloss_avg_bounds, dloss_avg_depths, dloss_avg_features
        )

        return dloss_bounds, dloss_depths, dloss_features


class PatchedMaximumReprojectionPooling(torch.autograd.Function):
    """
    Maximum pool the inputs, with respect to a culling map.
    Ignores the first input camera, as that is assumed to be the center camera.

    Arguments:
        bounds -- the depth lower bounds (B x N x 1 x H x W)
        depths -- the depth reprojections (B x N x 1 x H x W)
        features -- the feature reprojections (B x N x F x H x W)
        cull_mask -- the validity mask (B x N x 1 x H x W)

    Outputs:
        max_bounds -- the depth lower bounds (B x 1 x H x W)
        max_depths -- the depth reprojections (B x 1 x H x W)
        max_features -- the feature reprojections (B x F x H x W)
    """

    @staticmethod
    def forward(ctx, bounds, depths, features, cull_mask):
        B = features.shape[0]
        N = features.shape[1]
        F = features.shape[2]
        H = features.shape[3]
        W = features.shape[4]

        max_bounds = bounds.new_zeros((B, 1, H, W))
        max_depths = bounds.new_zeros((B, 1, H, W))
        max_features = bounds.new_zeros((B, F, H, W))

        MYTH.PatchedMaximumReprojectionPooling_updateOutput_gpu(
            bounds, depths, features,
            max_bounds, max_depths, max_features,
            cull_mask
        )

        ctx.save_for_backward(bounds, depths, cull_mask)
        return max_bounds, max_depths, max_features

    @staticmethod
    def backward(ctx, dloss_max_bounds, dloss_max_depths, dloss_max_features):
        bounds, depths, cull_mask = ctx.saved_variables
        B = dloss_max_features.shape[0]
        F = dloss_max_features.shape[1]
        H = dloss_max_features.shape[2]
        W = dloss_max_features.shape[3]
        N = cull_mask.shape[1]

        dloss_bounds = dloss_max_bounds.new_zeros((B, N, 1, H, W))
        dloss_depths = dloss_max_bounds.new_zeros((B, N, 1, H, W))
        dloss_features = dloss_max_bounds.new_zeros((B, N, F, H, W))

        MYTH.PatchedMaximumReprojectionPooling_updateGradInput_gpu(
            bounds, depths,
            dloss_bounds, dloss_depths, dloss_features,
            dloss_max_bounds, dloss_max_depths, dloss_max_features,
            cull_mask
        )

        return dloss_bounds, dloss_depths, dloss_features, None


class PatchedMinimumAbsReprojectionPooling(torch.autograd.Function):
    """
    Minimum pool the inputs' magnitudes, but keep the sign intact (with respect to a culling map).
    Ignores the first input camera, as that is assumed to be the center camera.

    Arguments:
        bounds -- the depth lower bounds (B x N x 1 x H x W)
        depths -- the depth reprojections (B x N x 1 x H x W)
        features -- the feature reprojections (B x N x F x H x W)
        cull_mask -- the validity mask (B x N x 1 x H x W)

    Outputs:
        min_bounds -- the depth lower bounds (B x 1 x H x W)
        min_depths -- the depth reprojections (B x 1 x H x W)
        min_features -- the feature reprojections (B x F x H x W)
    """

    @staticmethod
    def forward(ctx, bounds, depths, features, cull_mask):
        B = features.shape[0]
        N = features.shape[1]
        F = features.shape[2]
        H = features.shape[3]
        W = features.shape[4]

        max_bounds = bounds.new_zeros((B, 1, H, W))
        max_depths = bounds.new_zeros((B, 1, H, W))
        max_features = bounds.new_zeros((B, F, H, W))

        MYTH.PatchedMinimumAbsReprojectionPooling_updateOutput_gpu(
            bounds, depths, features,
            max_bounds, max_depths, max_features,
            cull_mask
        )

        ctx.save_for_backward(bounds, depths, cull_mask)
        return max_bounds, max_depths, max_features

    @staticmethod
    def backward(ctx, dloss_max_bounds, dloss_max_depths, dloss_max_features):
        bounds, depths, cull_mask = ctx.saved_variables
        B = dloss_max_features.shape[0]
        F = dloss_max_features.shape[1]
        H = dloss_max_features.shape[2]
        W = dloss_max_features.shape[3]
        N = cull_mask.shape[1]

        dloss_bounds = dloss_max_bounds.new_zeros((B, N, 1, H, W))
        dloss_depths = dloss_max_bounds.new_zeros((B, N, 1, H, W))
        dloss_features = dloss_max_bounds.new_zeros((B, N, F, H, W))

        MYTH.PatchedMinimumAbsReprojectionPooling_updateGradInput_gpu(
            bounds, depths,
            dloss_bounds, dloss_depths, dloss_features,
            dloss_max_bounds, dloss_max_depths, dloss_max_features,
            cull_mask
        )

        return dloss_bounds, dloss_depths, dloss_features, None
