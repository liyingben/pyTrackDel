# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import skimage.io as io
from skimage import measure, data, color
from shapely.geometry import Polygon

from Util import Util

from shapely.validation import explain_validity
from PIL import Image

pixScale = Util.pixScale
class ImgToPolygon(object):

    # 有点类似其它高级语言的构造函数
    def __init__(self, lon, lat):
        self.lon = lon
        self.lat = lat

    # 图片生成多边形
    def toPolygon(self, tileXy,imgPath):


        img = Image.open(imgPath)#打开图片
        newImg = Image.new("RGBA",(260,260),(255,255,255))
        newImg.paste(img,(2,2))
        newImg.save(imgPath)

        img = io.imread(imgPath)
        img = color.rgb2gray(img)  # 检测所有图形的轮廓
        contours = measure.find_contours(img, 0.8)  # 绘制轮廓

        polygonList = []
        for n, contour in enumerate(contours):
            pointList = []
            for xy in contour:
                pointList.append(self.pixelToLonlat(tileXy,xy[1]-2, xy[0]-2))
            if len(pointList) <4:
                continue
            p = Polygon(pointList)
            if not p.is_valid:
                p = p.buffer(0)
                assert p.is_valid, \
                    "Contour %r did not make valid polygon %s because %s" \
                    % (p, p.wkt, explain_validity(p))
            polygonList.append(p)
        return polygonList

    # 像素转换成经纬度
    def pixelToLonlat(self,tileXy, x, y):

        # if x == 0:
        #     x = -10
        # if x >= 256:
        #     x = 260
        # if y == 0:
        #     y = -10
        # if y >= 256:
        #     y = 260

        return (self.lon + (tileXy[0]*256.0 + x) / pixScale, self.lat + (tileXy[1]*256.0 + 256.0 - y) / pixScale)


# def main():
#     obj = ImgToPolygon(120, 40)
#     list = obj.toPolygon("/media/liyingben/g/pix/newPath/target/2017-09-08_3307_0_0.jpg")
#     for p in list:
#         print(p)
#
#
# main()
