# cython: language_level=3
import torch
cimport numpy as np
import numpy as np


cdef void _extract_patch_cython(np.ndarray output, np.ndarray image, np.ndarray pixels, int patch_radius):
    """
    Extract a given radius of pixels around a set of pixels in a given image.

    Arguments:
        output -- numpy array (size N x 3 x patch_size)
        image -- numpy array (size 3 x H x W)
        pixels -- numpy int array (size 2 x N)
        patch_radius -- int
    """
    cdef int N = pixels.shape[1]
    cdef int W = image.shape[2]
    cdef int H = image.shape[1]
    cdef int PD = patch_radius * 2 + 1

    cdef int n_x, n_y
    for p from 0 <= p < pixels.shape[1]:
        n_x = pixels[0, p]
        n_y = pixels[1, p]

        if n_x >= patch_radius and n_x < W - patch_radius and\
        n_y >= patch_radius and n_y < H - patch_radius:
            output[p, :, :, :] = image[
                :,
                n_y-patch_radius:n_y+patch_radius+1,
                n_x-patch_radius:n_x+patch_radius+1
            ]
    return

def extract_patch_cython(output, image, pixels, patch_radius):
    _extract_patch_cython(output, image, pixels, patch_radius)

def extract_patch_python_list_cat(image, pixels, patch_radius):
    """
    Extract a given radius of pixels around a set of pixels in a given image.

    Arguments:
        image -- pytorch Tensor (size 3 x H x W)
        pixels -- pytorch Tensor (size 2 x N)
        patch_radius -- int
    """
    patches = []
    image_width = image.shape[2]
    image_height = image.shape[1]

    for p in range(pixels.shape[1]):
        n_x = pixels[0, p]
        n_y = pixels[1, p]

        if n_x >= patch_radius and n_x < image_width - patch_radius and\
        n_y >= patch_radius and n_y < image_height - patch_radius:
            patches.append(image[
                :,
                n_y-patch_radius:n_y+patch_radius+1,
                n_x-patch_radius:n_x+patch_radius+1
            ].unsqueeze(0))
    return torch.cat(patches,0)


def extract_patch_python_prealloc(image, pixels, patch_radius):
    """
    Extract a given radius of pixels around a set of pixels in a given image.

    Arguments:
        image -- pytorch Tensor (size 3 x H x W)
        pixels -- numpy int array (size 2 x N)
        patch_radius -- int
    """
    N = pixels.shape[1]
    W = image.shape[2]
    H = image.shape[1]
    PD = patch_radius * 2 + 1
    patches = torch.empty((N, 3, PD, PD))

    for p in range(pixels.shape[1]):
        n_x = pixels[0, p]
        n_y = pixels[1, p]

        if n_x >= patch_radius and n_x < W - patch_radius and\
        n_y >= patch_radius and n_y < H - patch_radius:
            patches[p, :, :, :] = image[
                :,
                n_y-patch_radius:n_y+patch_radius+1,
                n_x-patch_radius:n_x+patch_radius+1
            ]
    return patches
