# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Util import Util
from FlowcountLineLayer import FlowcountLineLayer
import os
from DataBase import DataBase
from ImgToPolygon import ImgToPolygon
from FarmlandPolygonLayer import FarmlandPolygonLayer
from shapely.ops import cascaded_union
from shapely.wkt import dumps, loads
from shapely.geometry import MultiLineString

pixScale = Util.pixScale


class GenerateFarmland(object):
    # 有点类似其它高级语言的构造函数
    def __init__(self, flowcountLine, db, rootDir):
        self.rootDir = rootDir
        self.flowcountLine = flowcountLine
        self.keyList = []
        self.db = db

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
    def generate(self, date, moduleid):
        # 尝试将一个相交线写入到文件
        list = self.db.findAllLineByDateModuleid(date, moduleid)
        if len(list) == 0:
            return

        lineList = []
        for wkt in list:
            lineList.append(loads(wkt[2]))

        lines = MultiLineString(lineList)

        # print(lines.bounds)

        # box = self.boxByModuleidDate(date, moduleid)
        startLon = lines.bounds[0]
        startLat = lines.bounds[1]
        polygonList = []

        for geo in lines:
            # print("id=%d" % (geo[0]))
            box256 = self.createTileBox(startLon, startLat, geo.bounds[0], geo.bounds[1], geo.bounds[2], geo.bounds[3])
            lonlat = box256[0]
            tileXy = box256[1]
            key = str(date) + "_" + str(moduleid) + "_" + str(tileXy[0]) + "_" + str(tileXy[1])
            imgToPolygon = ImgToPolygon(startLon, startLat)

            if key not in self.keyList:
                filePath = self.getTrainTestImageByFileKey(key)
                if os.path.exists(filePath):
                    self.keyList.append(key)
                    list = imgToPolygon.toPolygon(tileXy, filePath)
                    polygonList = polygonList + list

        # 尝试将一天的地块多边形写入到文件
        polygonList = cascaded_union(polygonList)

        # for p in polygonList:
        #     self.flowcountLine.wirteIntersection(moduleid, str(date), lineList, p)
        # self.flowcountLine.save()

        return polygonList
    # 读取测试本地图片
    def getTrainTestImageByFileKey(self, key):
        path = self.rootDir + "images/" + key + "-outputs.png"

        return path


def main():
    # start = time.time()
    # date = '2017-09-07'
    # moduleid = 3575
    #
    db = DataBase(host='localhost',
                  port=3306,
                  user='root',
                  passwd='root',
                  db='flowcount',
                  table='flowcount_geometry_201710')
    flowcountLine = FlowcountLineLayer(
        "/media/liyingben/g/geoserver-2.11.0/data_dir/data/shapefiles/flowcountLine.properties")
    generateFarmland = GenerateFarmland(flowcountLine=flowcountLine,
                                        rootDir="/media/liyingben/g/pix/target_201710/",
                                        db=db)
    # generateFarmland.createTile(date, moduleid)
    writePolygon = FarmlandPolygonLayer(
        filePath="/media/liyingben/g/geoserver-2.11.0/data_dir/data/shapefiles/farmland_201710.properties")

    list = db.findDateModuleidList()
    for b in list:
        date = b[1]
        moduleid = b[0]
        polygonList = generateFarmland.generate(date, moduleid)
        if polygonList.geom_type == 'Polygon':
            writePolygon.wirte(moduleid, str(date), polygonList)
            continue

        for p in polygonList:
            writePolygon.wirte(moduleid, str(date), p)

    writePolygon.save()


main()
