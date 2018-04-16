# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Util import Util
from FlowcountLineLayer import FlowcountLineLayer
from GeoUtil import GeoUtil
import urllib
import os
from DataBase import DataBase
from shapely.wkt import dumps, loads
from shapely.geometry import MultiLineString
from PIL import Image
import ImageFilter as ImFilter
import ImageChops as ImChops
import ImageFont as ImFont
import ImageDraw as ImDraw
pixScale = Util.pixScale


class CreateSourceImgBydate(object):
    # 有点类似其它高级语言的构造函数
    def __init__(self, db, rootDir):

        self.rootDir = rootDir
        self.sourceKeyList = []
        self.db = db


    # 创建测试图片生成
    
    def createTileImg(self,list,startLon, startLat, bounds,imgPath):
        newImg = Image.new("RGBA", (256, 256), (255,255,255))
        draw = ImDraw.Draw(newImg)
        for geo in list:
    
            if geo.geom_type == 'LineString':
                line = self.geoToPix(startLon, startLat, bounds,geo)
                draw.line(line, (34,34,34),width=1)

        newImg.save(imgPath)



    def geoToPix(self,startLon, startLat, bounds,geo):
        list=geo.coords
    
        coords=[]
        for xy in list:
            px = int(((xy[0] - bounds[0]) * pixScale))
            py = 256-int(((xy[1] - bounds[1]) * pixScale))
            coords.append((px,py))
    
        return coords

    # 创建测试图片生成
    def createTileImgTarget(self,list,startLon, startLat, bounds,imgPath):
        newImg = Image.new("RGBA", (256, 256), (255,255,255))
        draw = ImDraw.Draw(newImg)
        for geo in list:
            if geo.geom_type == 'Polygon':
                polygon = self.geoToPixPolygon(startLon, startLat, bounds,geo)
                draw.polygon(polygon, fill=(34,34,34), outline=(34,34,34))
        newImg.save(imgPath)


    def geoToPixPolygon(self,startLon, startLat, bounds,geo):
        list= geo.exterior.coords

        coords=[]
        for xy in list:
            px = int(((xy[0] - bounds[0]) * pixScale))
            py = 256-int(((xy[1] - bounds[1]) * pixScale))
            coords.append((px,py))

        return coords
    
    # 创建训练资源图片
    def createTileBox(self, startLon, startLat, lon, lat, lon1, lat1):

        if lon < -180 or lon > 180 or lat < -90 or lat > 90:
            return

        tileX = int(((lon - startLon) * pixScale) / 256)
        px = int(((lon - startLon) * pixScale) % 256)
        tileY = int(((lat - startLat) * pixScale) / 256)
        py = int(((lat - startLat) * pixScale) % 256)

        box = [startLon + tileX * 256.0 / pixScale, startLat + tileY * 256.0 / pixScale,
               startLon + (tileX + 1) * 256.0 / pixScale, startLat + (tileY + 1) * 256.0 / pixScale]
        # print("box=%d,%d,%d,%d tileX,tileY=%d,%d px,py=%d,%d" % (box[0], box[1], box[2], box[3], tileX, tileY, px, py))

        return box, [tileX, tileY]

    # 创建训练数据图片
    def createTile(self, date, moduleid):


        # 尝试将一个相交线写入到文件
        list = self.db.findAllLineByDateModuleid(date, moduleid)
        if len(list) == 0:
            return

        # 尝试将一个相交线写入到文件
        targetList = self.db.findAllPolygonByDateModuleid(date, moduleid)
        if len(targetList) == 0:
            targets=[]
        else :
            targets =loads(targetList[0][2])

        lineList = []
        for wkt in list:
            lineList.append(loads(wkt[2]))

        lines = MultiLineString(lineList)

        print(lines.bounds)

        startLon = lines.bounds[0]
        startLat = lines.bounds[1]

        for geo in lines:
            # print("id=%d" % (geo[0]))
            box256 = self.createTileBox(startLon, startLat, geo.bounds[0], geo.bounds[1], geo.bounds[2], geo.bounds[3])
            bounds = box256[0]
            tileXy = box256[1]
            key = str(date) + "_" + str(moduleid) + "_" + str(tileXy[0]) + "_" + str(tileXy[1])

            if key not in self.sourceKeyList:
                self.createTileImg(lines,startLon, startLat, bounds,self.getSourcePath() + key + ".png")
                self.createTileImgTarget(targets,startLon, startLat, bounds,self.getTargetPath() + key + ".png")
                self.sourceKeyList.append(key)

    # 需要测试图片目录
    def getSourcePath(self):
        dir = self.rootDir + "/source/"
        self.mdDir(dir)
        return dir
    # 训练结果目录
    def getTargetPath(self):
        dir = self.rootDir + "/target/"
        self.mdDir(dir)
        return dir

    def mdDir(self, dir):
        isExists = os.path.exists(dir)
        # 判断结果
        if not isExists:
            # 如果不存在则创建目录
            # 创建目录操作函数
            os.makedirs(dir)


def main():
    # start = time.time()
    # date = '2017-09-07'
    # moduleid = 3575
    #
    db = DataBase(host='localhost', port=3306,
                  user='root', passwd='root',
                  db='flowcount', table='flowcount_geometry_201710')
    big = CreateSourceImgBydate(rootDir="/tmp/newPath",
                               db=db)
    # big.createTile(date, moduleid)

    list = db.findDateModuleidList()
    for b in list:
        date = b[1]
        moduleid = b[0]
        big.createTile(date, moduleid)


main()
