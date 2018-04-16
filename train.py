# coding=utf-8

import subprocess
from Layer import Layer
from DataBase import DataBase
import os



rootDir = os.path.abspath("") + "/"

wmsUrlTrainSource = "http://localhost:8080/geoserver/sf/wms?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image%2Fjpeg&STYLES&LAYERS=sf%3Aflowcount_geometry&tilesOrigin=-180%2C-90&WIDTH=256&HEIGHT=256&SRS=EPSG%3A900913&BBOX=";
wmsUrlTrainTarget = "http://localhost:8080/geoserver/sf/wms?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image%2Fjpeg&STYLES&LAYERS=sf%3AlandPolygon5&tilesOrigin=-180%2C-90&WIDTH=256&HEIGHT=256&SRS=EPSG%3A900913&BBOX=";
wmsUrlRoot = "http://123.56.66.184:9092/geoserver/ceshi/wms?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image%2Fjpeg&&styles=line&LAYERS=ceshi%3Aflowcount_geometry&tilesOrigin=-180%2C-90&WIDTH=256&HEIGHT=256&SRS=EPSG%3A900913&BBOX=";

layer = Layer(16,rootDir,wmsUrlRoot,wmsUrlTrainSource,wmsUrlTrainTarget)
dataBase = DataBase()

def run(cmd ):
    sub = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    sub.wait()


#将原始轨迹导入到flowcount_geometry
def updateToGeometry():
    print "UpdateToGeometry开始"
    dataBase.deleteAllByIdIsGreaterThan()
    dataBase.updateToGeometry()
    print "UpdateToGeometry结束"

# 创建训练数据图片
def createTrainTile():
    print "Create Image 开始"
    start = 0
    limit = 1000
    while True:
        list = dataBase.findTrainPage(start, limit)
        if len(list) ==0:
            break
        for geo in list:
            print "id=%d" %(geo[0])
            layer.createTrainSourceTile(geo[9], geo[10], geo[11], geo[12])
            layer.createTrainTargetTile(geo[9], geo[10], geo[11], geo[12])
        start += limit
    print "Create Image 结束"


#开始训练
def startTrain(epochs) :
    print "开始xunlian图片"
    run("python pix2pix.py --mode train --output_dir test/facades_BtoA_train --max_epochs 200 --input_dir /data/official/facades/train --which_direction BtoA --seed 0")
    print "DeepCmd结束开始处理图片"


#标记轨迹中要删除的数据
def deepLeaning():
    print "DeepLeaning开始"

    start = 0
    limit = 1000
    while True:
        list = dataBase.findPage(start, limit)
        for geo in list:
            print "id=" + geo[0]
            isDel = layer.isDelByTrainTestImage(geo)
            if isDel:
                dataBase.updateTypeById(1, geo[0])
                print "删除"
            else:
                dataBase.updateTypeById(0, geo[0])
                print "保留"

    print "DeepLeaning结束"


#标记轨迹中要删除的数据
def deepLeaningByDate( date):
    print "deepLeaningByDate开始"
    date = '2017-11-07'
    start = 0
    limit = 1000
    while True:
        list = dataBase.findPageByDate(start, limit, date)
        for geo in list:
            print "id=" + geo[0]
            isDel = layer.isDelByTrainTestImage(geo)
            if isDel:
                dataBase.updateTypeById(1, geo[0])
                print "删除"
            else:
                dataBase.updateTypeById(0, geo[0])
                print "保留"

        start += limit

    print "deepLeaningByDate结束"



layer.startTrainUnion()





