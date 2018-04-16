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

pixScale = Util.pixScale


class CreateTrainImgBydate(object):
    # 有点类似其它高级语言的构造函数
    def __init__(self, zoom, db, rootDir, source_layerName='ceshi%3Aflowcount_geometry_20179',
                 target_layerName='ceshi%3Aflytime_polygon_data'):
        self.zoom = zoom
        self.geoUtil = GeoUtil()
        self.rootDir = rootDir
        self.trainKeyList = []
        self.targetKeyList = []
        self.db = db
        self.source_layerName = source_layerName
        self.target_layerName = target_layerName
        # self.wmsUrlRootSource = "http://ceshi.farmfriend.com.cn/geoserver/ceshi/wms?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image%2Fpng&TRANSPARENT=true&STYLES&LAYERS=ceshi%3Aflowcount_geometry_20179&SRS=EPSG%3A4326"
        # self.wmsUrlRootTarget = "http://ceshi.farmfriend.com.cn/geoserver/ceshi/wms?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image%2Fpng&TRANSPARENT=true&STYLES=polygon_black&LAYERS=ceshi%3Aflytime_polygon_data&SRS=EPSG%3A4326"

        self.wmsUrlRootSource = "http://localhost:8080/geoserver/ceshi/wms?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image%2Fpng&TRANSPARENT=false&STYLES&LAYERS=" + self.source_layerName + "&SRS=EPSG%3A4326"
        self.wmsUrlRootTarget = "http://localhost:8080/geoserver/ceshi/wms?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image%2Fpng&TRANSPARENT=false&STYLES=polygon_black&LAYERS=" + self.target_layerName + "&SRS=EPSG%3A4326"

    # 创建测试图片生成
    def createTileUrl(self, urlRoot, lon, lat, lon1, lat1, moduleid, date):

        if lon < -180 or lon > 180 or lat < -90 or lat > 90:
            return
        width = int((lon1 - lon) * pixScale)
        heigh = int((lat1 - lat) * pixScale)
        width = 256
        heigh = 256
        print("moduleid=%d  date=%s width=%d heigh=%d box=%f,%f,%f,%f  " % (
            moduleid, date, width, heigh, lon, lat, lon1, lat1))
        url = urlRoot + "&WIDTH=" + str(width) + "&HEIGHT=" + str(heigh) + "&BBOX=" + str(lon) + "," + str(
            lat) + "," + str(lon1) + "," + str(lat1) + "&CQL_FILTER=moduleid=" + str(
            moduleid) + "%20AND%20startTime%20after%20" + date + "T00:00:00%20AND%20startTime%20BEFORE%20" + date + "T23:59:59Z"
        # print(url)
        return url

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
        print("Create Image 开始")

        # 尝试将一个相交线写入到文件
        list = self.db.findAllLineByDateModuleid(date, moduleid)
        if len(list) == 0:
            return
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
            lonlat = box256[0]
            tileXy = box256[1]
            key = str(date) + "_" + str(moduleid) + "_" + str(tileXy[0]) + "_" + str(tileXy[1])

            if key not in self.trainKeyList:
                url = self.createTileUrl(self.wmsUrlRootSource, lonlat[0], lonlat[1], lonlat[2], lonlat[3], moduleid,
                                         date)
                print(url)
                self.downImg(url, self.getTrainPath() + key + ".png")
                self.trainKeyList.append(key)
            if key not in self.targetKeyList:
                url = self.createTileUrl(self.wmsUrlRootTarget, lonlat[0], lonlat[1], lonlat[2], lonlat[3], moduleid,date)
                print(url)
                downPath = self.getTargetPath() + key + ".png"
                self.downImg(url, downPath)
                self.targetKeyList.append(key)

    def downImg(self, url, path):
        try:
            urllib.urlretrieve(url, path)  # 下载图片
        except:
            print("Error： " + url)

    # 需要测试图片目录
    def getTrainPath(self):
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
    big = CreateTrainImgBydate(zoom=10, rootDir="/tmp/newPath",
                               db=db, source_layerName='ceshi%3Aflowcount_geometry_201710',
                               target_layerName='ceshi%3Aflytime_polygon_data')
    # big.createTile(date, moduleid)

    list = db.findDateModuleidList()
    for b in list:
        date = b[1]
        moduleid = b[0]
        big.createTile(date, moduleid)


main()
