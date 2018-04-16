# coding=utf-8


import math

# 各地图API坐标系统比较与转换
# WGS84坐标系：即地球坐标系，国际上通用的坐标系。设备一般包含GPS芯片或者北斗芯片获取的经纬度为WGS84地理坐标系,
# 谷歌地图采用的是WGS84地理坐标系（中国范围除外）
# GCJ02坐标系：即火星坐标系，是由中国国家测绘局制订的地理信息系统的坐标系统。由WGS84坐标系经加密后的坐标系。
# 谷歌中国地图和搜搜中国地图采用的是GCJ02地理坐标系 BD09坐标系：即百度坐标系，GCJ02坐标系经加密后的坐标系
# 搜狗坐标系、图吧坐标系等，估计也是在GCJ02基础上加密而成的。

BAIDU_LBS_TYPE = "bd09ll"
pi = 3.1415926535897932384626
a = 6378245.0
ee = 0.00669342162296594323

class GeoUtil():

    # 84 to 火星坐标系 (GCJ-02) World Geodetic System ==> Mars Geodetic System
    # lat
    # lon
    def gps84_To_Gcj02(self,lat, lon):
        if self.outOfChina(lat, lon):
            return

        dLat = self.transformLat(lon - 105.0, lat - 35.0)
        dLon = self.transformLon(lon - 105.0, lat - 35.0)
        radLat = lat / 180.0 * pi
        magic = math.sin(radLat)
        magic = 1 - ee * magic * magic
        sqrtMagic = math.sqrt(magic)
        dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * pi)
        dLon = (dLon * 180.0) / (a / sqrtMagic * math.cos(radLat) * pi)
        mgLat = lat + dLat
        mgLon = lon + dLon
        return {"lat": mgLat, "lon": mgLon}


    # 火星坐标系 (GCJ-02) to 84 * * @param lon * @param lat * @return
    def gcj_To_Gps84(self,lat, lon):
        gps = self.transform(lat, lon)
        lontitude = lon * 2 - gps['lon']
        latitude = lat * 2 - gps['lat']
        return {"lat": latitude, "lon": lontitude}


    # 火星坐标系 (GCJ-02) 与百度坐标系 (BD-09) 的转换算法 将 GCJ-02 坐标转换成 BD-09 坐标
    def gcj02_To_Bd09(self,gg_lat, gg_lon):
        x = gg_lon
        y = gg_lat
        z = math.sqrt(x * x + y * y) + 0.00002 * math.sin(y * pi)
        theta = math.atan2(y, x) + 0.000003 * math.cos(x * pi)
        bd_lon = z * math.cos(theta) + 0.0065
        bd_lat = z * math.sin(theta) + 0.006
        return {"lat": bd_lat, "lon": bd_lon}


    # 火星坐标系 (GCJ-02) 与百度坐标系 (BD-09) 的转换算法 * * 将 BD-09 坐标转换成GCJ-02 坐标 * * @param
    def bd09_To_Gcj02(self,bd_lat, bd_lon):
        x = bd_lon - 0.0065
        y = bd_lat - 0.006
        z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * pi)
        theta = math.atan2(y, x) - 0.000003 * math.cos(x * pi)
        gg_lon = z * math.cos(theta)
        gg_lat = z * math.sin(theta)
        return {"lat": gg_lat, "lon": gg_lon}


    # (BD-09)-->84
    def bd09_To_Gps84(self,bd_lat, bd_lon):
        gcj02 = self.bd09_To_Gcj02(bd_lat, bd_lon)
        map84 = self.gcj_To_Gps84(gcj02['lat'], gcj02['lon'])
        return map84


    def transform(self,lat, lon):
        if self.outOfChina(lat, lon):
            return {"lat": lat, "lon": lon}

        dLat = self.transformLat(lon - 105.0, lat - 35.0)
        dLon = self.transformLon(lon - 105.0, lat - 35.0)
        radLat = lat / 180.0 * pi
        magic = math.sin(radLat)
        magic = 1 - ee * magic * magic
        sqrtMagic = math.sqrt(magic)
        dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * pi)
        dLon = (dLon * 180.0) / (a / sqrtMagic * math.cos(radLat) * pi)
        mgLat = lat + dLat
        mgLon = lon + dLon
        return {"lat": mgLat, "lon": mgLon}


    def transformLat(self,x, y):
        ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(y * pi) + 40.0 * math.sin(y / 3.0 * pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(y / 12.0 * pi) + 320 * math.sin(y * pi / 30.0)) * 2.0 / 3.0
        return ret


    def transformLon(self,x, y):
        ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(x * pi) + 40.0 * math.sin(x / 3.0 * pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(x / 12.0 * pi) + 300.0 * math.sin(x / 30.0 * pi)) * 2.0 / 3.0
        return ret


    def outOfChina(self,lat, lon):
        if lon < 72.004 or lon > 137.8347:
            return True
        if lat < 0.8293 or lat > 55.8271:
            return True
        return False




    # 经纬度转墨卡托
    # 经度(lon)，纬度(lat)
    def lonLat2Mercator(self, lon, lat):
        xy = []
        x = lon * 20037508.342789 / 180
        y = math.log(math.tan((90 + lat) * pi / 360)) / (pi / 180)
        y = y * 20037508.34789 / 180
        xy.append(x)
        xy.append(y)
        return xy


    # 墨卡托转经纬度
    def Mercator2lonLat(self,mercatorX, mercatorY):
        xy = []
        x = mercatorX / 20037508.34 * 180
        y = mercatorY / 20037508.34 * 180
        y = 180 / pi * (2 * math.atan(math.exp(y * pi / 180)) - pi / 2)
        xy.append(x)
        xy.append(y)
        return xy
