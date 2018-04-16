# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import skimage.io as io
from skimage import measure, data, color
from shapely.geometry import Polygon



class Util(object):
    # pixScale = 23300.0
    pixScale = 20000.0