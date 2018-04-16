# coding=utf-8

from GeoUtil import GeoUtil
import urllib
import os
import numpy as np
from PIL import Image
from datetime import datetime
import subprocess

zoom = 16
mapwidth = 20037508.34

espg900913 = [
    156543.03390625,
    78271.516953125,
    39135.7584765625,
    19567.87923828125,
    9783.939619140625,
    4891.9698095703125,
    2445.9849047851562,
    1222.9924523925781,
    611.4962261962891,
    305.74811309814453,
    152.87405654907226,
    76.43702827453613,
    38.218514137268066,
    19.109257068634033,
    9.554628534317017,
    4.777314267158508,
    2.388657133579254,
    1.194328566789627,
    0.5971642833948135,
    0.2985821416974068,
    0.1492910708487034,
    0.0746455354243517,
    0.0373227677121758,
    0.0186613838560879,
    0.009330691928044,
    0.004665345964022,
    0.002332672982011,
    0.0011663364910055,
    0.0005831682455027,
    0.0002915841227514,
    0.0001457920613757
]


pixDirRoot = "/home/liyingben/pix2pix-tensorflow"
wmsUrlRoot = "http://localhost:8080/geoserver/sf/wms?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image%2Fjpeg&STYLES&LAYERS=sf%3Aflowcount_geometry&tilesOrigin=-180%2C-90&WIDTH=256&HEIGHT=256&SRS=EPSG%3A900913&BBOX="
wmsUrlTrainSource = "http://localhost:8080/geoserver/sf/wms?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image%2Fjpeg&STYLES&LAYERS=sf%3Aflowcount_geometry&tilesOrigin=-180%2C-90&WIDTH=256&HEIGHT=256&SRS=EPSG%3A900913&BBOX="
wmsUrlTrainTarget = "http://localhost:8080/geoserver/sf/wms?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image%2Fjpeg&STYLES&LAYERS=sf%3Aflowcount_geometry&tilesOrigin=-180%2C-90&WIDTH=256&HEIGHT=256&SRS=EPSG%3A900913&BBOX="


class Layer(object):
    # 有点类似其它高级语言的构造函数
    def __init__(self, zoom, rootDir, wmsUrlRoot, wmsUrlTrainSource, wmsUrlTrainTarget):
        self.zoom = zoom
        self.rootDir = rootDir
        self.wmsUrlRoot = wmsUrlRoot
        self.wmsUrlTrainSource = wmsUrlTrainSource
        self.wmsUrlTrainTarget = wmsUrlTrainTarget
        self.resolution = espg900913[zoom]
        self.geoUtil = GeoUtil()

    # xun lian tu pian xia zai
    trainSourceTilekeyListUrl = []

    # 创建训练资源图片
    def createTrainSourceTile(self, lon, lat, lon1, lat1):

        if lon < -180 or lon > 180 or lat < -90 or lat > 90:
            return

        xyList = []
        xyList.append(self.geoUtil.lonLat2Mercator(lon, lat))
        xyList.append(self.geoUtil.lonLat2Mercator(lon1, lat1))
        for xy in xyList:
            tileX = int(((xy[0] + mapwidth) / self.resolution) / 256)
            px = int(((xy[0] + mapwidth) / self.resolution) % 256)
            tileY =int( ((mapwidth - xy[1]) / self.resolution) / 256)
            py =int( ((mapwidth - xy[1]) / self.resolution) % 256)

            box = [tileX * self.resolution * 256 - mapwidth, mapwidth - (tileY + 1) * self.resolution * 256,
                   (tileX + 1) * self.resolution * 256 - mapwidth, mapwidth - tileY * self.resolution * 256]
            print "box=%d,%d,%d,%d" % (box[0], box[1], box[2], box[3])
            print "tileX,tileY=%d,%d" %(tileX, tileY)
            print "px,py=%d,%d" %(px, py)

            url = wmsUrlTrainSource + str(box[0]) + "," + str(box[1]) + "," + str(box[2]) + "," + str(box[3])

            print url
            key = str(tileX) + "_" + str(tileY)
            if key not in self.trainSourceTilekeyListUrl:
                self.downImg(url, self.getTrainSourcePath() + key + ".jpg")
                self.trainSourceTilekeyListUrl.append(key)

    trainTargetTilekeyListUrl = []

    # 创建训练目标图片
    def createTrainTargetTile(self, lon, lat, lon1, lat1):

        if lon < -180 or lon > 180 or lat < -90 or lat > 90:
            return

        xyList = []
        xyList.append(self.geoUtil.lonLat2Mercator(lon, lat))
        xyList.append(self.geoUtil.lonLat2Mercator(lon1, lat1))
        for xy in xyList:
            tileX = int(((xy[0] + mapwidth) / self.resolution)/ 256)
            px = int(((xy[0] + mapwidth) / self.resolution)% 256)
            tileY = int(((mapwidth - xy[1]) / self.resolution) / 256)
            py = int(((mapwidth - xy[1]) / self.resolution) % 256)

            box = [tileX * self.resolution * 256 - mapwidth, mapwidth - (tileY + 1) * self.resolution * 256,
                   (tileX + 1) * self.resolution * 256 - mapwidth, mapwidth - tileY * self.resolution * 256]
            print "box=%d,%d,%d,%d" % (box[0], box[1], box[2], box[3])
            print "tileX,tileY=%d,%d" %(tileX, tileY)
            print "px,py=%d,%d" %(px, py)
            url = wmsUrlTrainTarget + str(box[0]) + "," + str(box[1]) + "," + str(box[2]) + "," + str(box[3])

            print url
            key = str(tileX) + "_" + str(tileY)

            if key not in self.trainSourceTilekeyListUrl:
                self.downImg(url, self.getTrainTargetPath() + key + ".jpg")
                self.trainTargetTilekeyListUrl.append(key)

    testTilekeyListUrl = []

    # 创建测试图片生成
    def createTestTile(self, lon, lat, lon1, lat1, moduleid, date):

        if lon < -180 or lon > 180 or lat < -90 or lat > 90:
            return

        xyList = []
        xyList.add(self.geoUtil.lonLat2Mercator(lon, lat))
        xyList.add(self.geoUtil.lonLat2Mercator(lon1, lat1))
        for xy in xyList:
            tileX = int(((xy[0] + mapwidth) / self.resolution).intValue() / 256)
            px = int(((xy[0] + mapwidth) / self.resolution).intValue() % 256)
            tileY = int(((mapwidth - xy[1]) / self.resolution).intValue() / 256)
            py = int(((mapwidth - xy[1]) / self.resolution).intValue() % 256)

            box = [tileX * self.resolution * 256 - mapwidth, mapwidth - (tileY + 1) * self.resolution * 256,
                   (tileX + 1) * self.resolution * 256 - mapwidth, mapwidth - tileY * self.resolution * 256]
            print "box=" + box[0] + "," + box[1] + "," + box[2] + "," + box[3]
            print "tileX,tileY=" + tileX + "," + tileY
            print "px,py=" + px + "," + py
            date = datetime.strptime(date, '%Y-%m-%d')
            url = wmsUrlRoot + box[0] + "," + box[1] + "," + box[2] + "," + box[
                3] + "&CQL_FILTER=moduleid=" + moduleid + "%20AND%20createtime%20after%20" + date + "T00:00:00%20AND%20createtime%20BEFORE%20" + date + "T23:59:59Z"

            print url
            key = tileX + "_" + tileY

            if key not in self.testTilekeyListUrl:
                self.downImg(url, self.getTestPath() + key + ".jpg")
                self.testTilekeyListUrl.append(key)

    keyList = []
    keyMap = {}

    def isDelByTrainTestImage(self, lon, lat, lon1, lat1, moduleid, date):
        xyList = []
        xyList.add(self.geoUtil.lonLat2Mercator(lon1, lat1))
        xyList.add(self.geoUtil.lonLat2Mercator(lon1, lat1))

        for xy in xyList:
            tileX = int(((xy[0] + mapwidth) / self.resolution).intValue() / 256)
            px = int(((xy[0] + mapwidth) / self.resolution).intValue() % 256)
            tileY = int(((mapwidth - xy[1]) / self.resolution).intValue() / 256)
            py = int(((mapwidth - xy[1]) / self.resolution).intValue() % 256)

        box = [tileX * self.resolution * 256 - mapwidth, mapwidth - (tileY + 1) * self.resolution * 256,
               (tileX + 1) * self.resolution * 256 - mapwidth, mapwidth - tileY * self.resolution * 256]
        print "box=" + box[0] + "," + box[1] + "," + box[2] + "," + box[3]
        print "tileX,tileY=" + tileX + "," + tileY
        print "px,py=" + px + "," + py
        date = datetime.strptime(date, '%Y-%m-%d')
        url = wmsUrlRoot + box[0] + "," + box[1] + "," + box[2] + "," + box[
            3] + "&CQL_FILTER=moduleid=" + moduleid + "%20AND%20createtime%20after%20" + date + "T00:00:00%20AND%20createtime%20BEFORE%20" + date + "T23:59:59Z"

        print url
        key = tileX + "_" + tileY + "_" + moduleid + "_" + date

        if key not in self.keyMap:
            bi = self.getTrainTestImageByFileKey(key)
            if bi is None:
                return False
            self.keyList.addLast(key)
            self.keyMap.put(key, bi)
            self.checKeyList()
        else:
            bi = self.keyMap.get(key)
            rgb = self.getImagePixel(bi, px, py)
            if (rgb[0] + rgb[1] + rgb[2]) < 750:
                return False
        return True

    # 读取一张图片的RGB值
    def getImagePixel(self, bi, i, j):
        rgb = []
        rgb = bi[i, j]  # 下面三行代码将一个数字转换为RGB数字

        print "i=" + i + ",j=" + j + ":(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + ")"
        return [rgb[0], rgb[1], rgb[2]]

    # 读取测试本地图片
    def getTrainTestImageByFileKey(self, key):
        path = self.getTestOutPath() + "/images/" + key + "-outputs.png"
        print path
        return np.array(Image.open(path))

    # 释放缓存
    def checKeyList(self):
        if len(self.keyList) > 1000:
            key = self.keyList.pollFirst()
            self.keyMap.remove(key)

    def downImg(self, url, path):
        try:
            urllib.urlretrieve(url, path)  # 下载图片
        except:
            print("Error： " + url)

    # 需要测试图片目录
    def getTestPath(self):
        dir = self.rootDir + "/test/"
        self.mdDir(dir)
        return dir

    # 需要测试图片目录
    def getTestUnionPath(self):
        dir = self.rootDir + "/test_union/"
        self.mdDir(dir)
        return dir

    # 测试结果目录
    def getTestOutPath(self):
        dir = self.rootDir + "/test_out/"
        self.mdDir(dir)
        return dir

    # 训练结果目录
    def getTrainPath(self):
        dir = self.rootDir + "/train/"
        self.mdDir(dir)
        return dir

    # 训练结果目录
    def getTrainSourcePath(self):
        dir = self.rootDir + "/source/"
        self.mdDir(dir)
        return dir

    # 训练结果目录
    def getTrainTargetPath(self):
        dir = self.rootDir + "/target/"
        self.mdDir(dir)
        return dir

    # 训练结果目录
    def getTrainUnionPath(self):
        dir = self.rootDir + "/train_union/"
        self.mdDir(dir)
        return dir

    def mdDir(self,dir):
        isExists=os.path.exists(dir)
        # 判断结果
        if not isExists:
            # 如果不存在则创建目录
            # 创建目录操作函数
            os.makedirs(dir)


    def startTrainUnion(self) :
        print "开始startTrainUnion图片"
        self.run("python process.py --operation combine --input_dir "+self.getTrainSourcePath()+" --b_dir "+self.getTrainTargetPath()+" --output_dir  "+self.getTrainUnionPath())
        print "startTrainUnion结束"

    def run(self,cmd ):
        sub = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        sub.wait()
