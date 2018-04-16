# coding=utf-8

from process import combine
import argparse
import os
import tempfile
import subprocess
import tensorflow as tf
import numpy as np
import tfimage as im
import threading
import time
import multiprocessing
src = im.load("")


def combine(src, src_path,b_dir):
    if b_dir is None:
        raise Exception("missing b_dir")

    # find corresponding file in b_dir, could have a different extension
    basename, _ = os.path.splitext(os.path.basename(src_path))
    for ext in [".png", ".jpg"]:
        sibling_path = os.path.join(b_dir, basename + ext)
        if os.path.exists(sibling_path):
            sibling = im.load(sibling_path)
            break
    else:
        raise Exception("could not find sibling image for " + src_path)

    # make sure that dimensions are correct
    height, width, _ = src.shape
    if height != sibling.shape[0] or width != sibling.shape[1]:
        raise Exception("differing sizes")

    # convert both images to RGB if necessary
    if src.shape[2] == 1:
        src = im.grayscale_to_rgb(images=src)

    if sibling.shape[2] == 1:
        sibling = im.grayscale_to_rgb(images=sibling)

    # remove alpha channel
    if src.shape[2] == 4:
        src = src[:,:,:3]

    if sibling.shape[2] == 4:
        sibling = sibling[:,:,:3]

    return np.concatenate([src, sibling], axis=1)