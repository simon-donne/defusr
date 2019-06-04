"""
Helper functions to do with the file system.
"""
import os

def ensure_dir(path):
    """
    Makes sure a folder (and all of its parents) exists.
    """
    if not os.path.exists(path):
        os.makedirs(path)
