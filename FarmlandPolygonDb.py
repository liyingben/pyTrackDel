# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import skimage.io as io
from skimage import measure, data, color
from shapely.geometry import Polygon

class FarmlandPolygonDb(object):
    # 有点类似其它高级语言的构造函数
    def __init__(self, db):
        self.db = db


    # 写多边形
    def wirte(self, moduleid, date, polygon):
        self.f.write('stations.%d=%d|%s|%s\n' % (moduleid,moduleid, date, polygon))
    # 写多边形
    def wirteList(self,moduleid, date, polygons):
        for polygon in polygons:
            self.wirte(moduleid, date, polygon)



# def main():
#     obj = WritePolygon("")
#     obj.wirte(3046, "2017-12-03", "POLYGON ((120.0085836909871 40.00504291845493, 120.0085622317597 40.00502145922747, 120.0085836909871 40.005, 120.0086051502146 40.00502145922747, 120.0085836909871 40.00504291845493))")
#     obj.save()
#
#
# main()
