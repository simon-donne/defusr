"""
Utilities for faster debugging.
"""

import cv2
import os
from utils.tmp import TemporaryFolder
from utils.depth_map_visualization import color_depth_map, color_error_image, color_trust_image

folder = TemporaryFolder(name="debug_images")

def save_gray_tensor(data, index=0):
    # index with zeros until we have a 2D array
    while len(data.shape) > 2:
        data = data[0]
    
    # now detach, get numpy array
    data = data.detach().cpu().numpy()

    if data.max() <= 1 and data.min() >= 0:
        data = data * 255
    
    out_path = os.path.join(str(folder), "%05d.png" % index)
    cv2.imwrite(out_path, data)
    print("Saved debug image to " + out_path)

def save_depth_tensor(data, index=0, scale=None, mask=None):
    # index with zeros until we have a 2D array
    while len(data.shape) > 2:
        data = data[0]
    
    # now detach, get numpy array
    data = data.detach().cpu().numpy()

    colored_data = color_depth_map(data, scale=scale, mask=mask.detach().cpu().numpy() if mask is not None else None)
    
    out_path = os.path.join(str(folder), "%05d.png" % index)
    cv2.imwrite(out_path, colored_data)
    print("Saved debug image to " + out_path)

def save_trust_tensor(data, index=0, mask=None):
    # index with zeros until we have a 2D array
    while len(data.shape) > 2:
        data = data[0]
    
    # now detach, get numpy array
    data = data.detach().cpu().numpy()

    colored_data = color_trust_image(data, mask=mask.detach().cpu().numpy() if mask is not None else None)
    
    out_path = os.path.join(str(folder), "%05d.png" % index)
    cv2.imwrite(out_path, colored_data)
    print("Saved debug image to " + out_path)

def save_error_tensor(data, index=0, scale=1, mask=None):
    # index with zeros until we have a 2D array
    while len(data.shape) > 2:
        data = data[0]
    
    # now detach, get numpy array
    data = data.detach().cpu().numpy()

    colored_data = color_error_image(data, scale=scale, mask=mask.detach().cpu().numpy() if mask is not None else None)
    
    out_path = os.path.join(str(folder), "%05d.png" % index)
    cv2.imwrite(out_path, colored_data)
    print("Saved debug image to " + out_path)

def save_color_tensor(data, index=0):
    # index with zeros until we have a 3D array
    while len(data.shape) > 3:
        data = data[0]
    
    # now detach, get numpy array
    data = data.detach().cpu().numpy()

    # also transpose because OpenCV
    data = data.transpose(1,2,0)

    if data.max() <= 1 and data.min() >= 0:
        data = data * 255
    
    out_path = os.path.join(str(folder), "%05d.png" % index)
    cv2.imwrite(out_path, data)
    print("Saved debug image to " + out_path)

def save_gray_numpy(data, index=0):
    # index with zeros until we have a 2D array
    while len(data.shape) > 2:
        data = data[0]

    if data.max() <= 1 and data.min() >= 0:
        data = data * 255
    
    out_path = os.path.join(str(folder), "%05d.png" % index)
    cv2.imwrite(out_path, data)
    print("Saved debug image to " + out_path)

def save_depth_numpy(data, index=0, scale=None):
    # index with zeros until we have a 2D array
    while len(data.shape) > 2:
        data = data[0]

    colored_data = color_depth_map(data, scale=scale)
    
    out_path = os.path.join(str(folder), "%05d.png" % index)
    cv2.imwrite(out_path, colored_data)
    print("Saved debug image to " + out_path)

def save_trust_numpy(data, index=0):
    # index with zeros until we have a 2D array
    while len(data.shape) > 2:
        data = data[0]
    
    colored_data = color_trust_image(data)
    
    out_path = os.path.join(str(folder), "%05d.png" % index)
    cv2.imwrite(out_path, colored_data)
    print("Saved debug image to " + out_path)

def save_error_numpy(data, index=0, scale=1):
    # index with zeros until we have a 2D array
    while len(data.shape) > 2:
        data = data[0]
    
    colored_data = color_error_image(data, scale=scale)
    
    out_path = os.path.join(str(folder), "%05d.png" % index)
    cv2.imwrite(out_path, colored_data)
    print("Saved debug image to " + out_path)

def save_color_numpy(data, index=0):
    # index with zeros until we have a 3D array
    while len(data.shape) > 3:
        data = data[0]
    
    # also transpose because OpenCV
    if data.shape[2] > 4:
        data = data.transpose(1,2,0)

    if data.max() <= 1 and data.min() >= 0:
        data = data * 255
    
    out_path = os.path.join(str(folder), "%05d.png" % index)
    cv2.imwrite(out_path, data)
    print("Saved debug image to " + out_path)
