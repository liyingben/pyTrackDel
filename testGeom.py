#!/usr/bin/env python # -*- coding: utf-8 -*-
import skimage.io as io
import matplotlib.pyplot as plt
from skimage import measure, data, color
from shapely.geometry import Polygon

img = io.imread("test.png")
img = color.rgb2gray(img)  # 检测所有图形的轮廓
contours = measure.find_contours(img, 0.5)  # 绘制轮廓
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 8))
ax0.imshow(img, plt.cm.gray)
ax1.imshow(img, plt.cm.gray)
f = open('/tmp/test.txt', 'w')
f.write("_=id:Integer,location:Geometry:srid=4326\n")
i = 1
for n, contour in enumerate(contours):
    ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
    xyList = []
    polygonList = []
    for xy in contour:
        xyList.append(str(xy[1]) + " " + str(xy[0]))
        polygonList.append((xy[1], xy[0]))
    polygon = "POLYGON ((" + ",".join(xyList) + "))"
    p = Polygon(polygonList)
    f.write('stations.%d=%d|%s\n' % (i, i, p))
    print p.area
    i = i + 1
    print p
f.close()
ax1.axis('image')
ax1.set_xticks([])
ax1.set_yticks([])
plt.show()
