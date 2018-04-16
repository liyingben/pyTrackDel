# coding=utf-8

import subprocess

from CreateSourceImgBydate import CreateSourceImgBydate
from FarmlandPolygonDb import FarmlandPolygonDb
from FarmlandPolygonLayer import FarmlandPolygonLayer
from GenerateFarmland import GenerateFarmland
from Layer import Layer
from DataBase import DataBase
import urllib
import os
import numpy as np
from PIL import Image
from datetime import datetime

rootDir = os.path.abspath("") + "/"
rootDir ="/media/liyingben/g/pix/imgToPolygon"

db = DataBase(host='localhost',
              port=3306,
              user='root',
              passwd='root',
              db='flowcount',
              table='flowcount_geometry_201710')


def run(cmd):
    sub = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    sub.wait()


def startTrainUnion():
    print "开始startTrainUnion图片"
    run(
        "python process.py --operation combine --input_dir " + getTrainSourcePath() + " --b_dir " + getTrainTargetPath() + " --output_dir  " + getTrainUnionPath())
    print "startTrainUnion结束"


def startTestUnion():
    print "开始startTestUnion图片"
    run(
        "python process.py --operation combine --input_dir " + getTestPath() + " --b_dir " + getTestPath() + " --output_dir  " + getTestUnionPath())
    print "startTestUnion结束"


def startTest():
    print "开始startTest图片"
    run(
        "python pix2pix.py --mode test --output_dir " + getTestOutPath() + " --input_dir " + getTestUnionPath() + " --checkpoint " + getTrainPath())
    print "startTest结束"


def createSourceTile():
    # start = time.time()
    # date = '2017-09-07'
    # moduleid = 3575
    #

    big = CreateSourceImgBydate(rootDir=rootDir, db=db)
    # big.createTile(date, moduleid)

    list = db.findDateModuleidList()
    for b in list:
        date = b[1]
        moduleid = b[0]
        big.createTile(date, moduleid)


def generatePolygon():
    # start = time.time()
    # date = '2017-09-07'
    # moduleid = 3575
    #


    generateFarmland = GenerateFarmland(rootDir=getTestOutPath(), db=db)
    # generateFarmland.createTile(date, moduleid)
    writePolygon = FarmlandPolygonDb(db=db)

    list = db.findDateModuleidList()
    for b in list:
        date = b[1]
        moduleid = b[0]
        polygonList = generateFarmland.generate(date, moduleid)
        if polygonList.geom_type == 'Polygon':
            writePolygon.insertPolygon(moduleid, str(date), polygonList)
            continue

        for p in polygonList:
            writePolygon.insertPolygon(moduleid, str(date), p)


# 需要测试图片目录
def getTestPath():
    dir = rootDir + "/test/"
    mdDir(dir)
    return dir


# 需要测试图片目录
def getTestUnionPath():
    dir = rootDir + "/test_union/"
    mdDir(dir)
    return dir


# 测试结果目录
def getTestOutPath():
    dir = rootDir + "/test_out/"
    mdDir(dir)
    return dir


# 训练结果目录
def getTrainPath():
    dir = rootDir + "/train/"
    mdDir(dir)
    return dir


# 训练结果目录
def getTrainSourcePath():
    dir = rootDir + "/source/"
    mdDir(dir)
    return dir


# 训练结果目录
def getTrainTargetPath():
    dir = rootDir + "/target/"
    mdDir(dir)
    return dir


# 训练结果目录
def getTrainUnionPath():
    dir = rootDir + "/train_union/"
    mdDir(dir)
    return dir


def mdDir(dir):
    isExists = os.path.exists(dir)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(dir)


createSourceTile()
startTestUnion()
startTest()
generatePolygon()
