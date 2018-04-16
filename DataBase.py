# coding=utf-8
import re
import urllib2
import urllib
import os

# 导入MySQLdb模块
import MySQLdb


class DataBase(object):

    # 有点类似其它高级语言的构造函数
    def __init__(self, host='localhost'  # 主机名
                 , port=3306  # 端口
                 , user='root'  # 用户名
                 , passwd='root'  # 密码
                 , db='flowcount',table="flowcount_geometry_20179"):
        # 建立与数据库的连接
        self.conn = MySQLdb.connect(host=host  # 主机名
                                    , port=port  # 端口
                                    , user=user  # 用户名
                                    , passwd=passwd  # 密码
                                    , db=db  # 数据库名
                                    )
        self.table=table
        # 也可以简写成conn1=MySQLdb.connect('localhost','root','admin','test123')
        # 创建游标
        # cur1 = conn1.cursor()
        # 对游标对象使用execute方法就可以执行普通的SQL语句
        # 做一次查询
        # cur1.execute('SELECT id, moduleid,createtime,receivetime,packetnumber,powertimes,curflow,sumflow,version,  st_x(StartPoint(geom)) AS lon,st_y(StartPoint(geom)) as lat, st_x(EndPoint(geom)) AS lon1,st_y(EndPoint(geom)) as lat1,0 as type FROM flowcount_20175 where IsEmpty(geom) =0 order by id limit 100  ')
        # 通过fetchall方法获取全部查询结果
        # result1 = cur1.fetchall()
        # print result1
        # 通过fetchone方法获取一条查询结果
        # result2 = cur1.fetchone()
        # print result2
        # 关闭游标
        # cur1.close()
        # 断开数据库连接
        # conn1.close()

    def findTrainPage(self, start, limit):
        # 创建游标
        cur = self.conn.cursor()

        # 对游标对象使用execute方法就可以执行普通的SQL语句
        # 做一次查询
        cur.execute(
            ' SELECT id, moduleid,createtime,receivetime,packetnumber,powertimes,curflow,sumflow,version,  st_x(StartPoint(geom)) AS lon,st_y(StartPoint(geom)) as lat, st_x(EndPoint(geom)) AS lon1,st_y(EndPoint(geom)) as lat1,0 as type FROM flowcount_geometry where IsEmpty(geom) =0 order by id limit %d,%d '
            % (int(start), int(limit)))

        # 通过fetchall方法获取全部查询结果
        result = cur.fetchall()
        # 关闭游标
        cur.close()
        # print result
        return result

    def updateTypeById(self, type, id):
        # 创建游标
        cur = self.conn.cursor()

        # 对游标对象使用execute方法就可以执行普通的SQL语句
        # 做一次查询
        cur.execute(' update flowcount_geometry u set u.type =%d where u.id =%d' % (int(type), int(id)))

        # 通过fetchall方法获取全部查询结果
        result = cur.fetchone()
        # result = cur.fetchall()
        # 关闭游标
        cur.close()
        # print result
        return result

    def deleteAllByIdIsGreaterThan(self):
        # 创建游标
        cur = self.conn.cursor()

        # 对游标对象使用execute方法就可以执行普通的SQL语句
        # 做一次查询
        cur.execute(' DELETE FROM flowcount.flowcount_geometry  WHERE id>=0 ')

        # 关闭游标
        cur.close()

    def findPage(self, start, limit):
        # 创建游标
        cur = self.conn.cursor()

        # 对游标对象使用execute方法就可以执行普通的SQL语句
        # 做一次查询
        cur.execute(
            ' SELECT id, moduleid,createtime,receivetime,packetnumber,powertimes,curflow,sumflow,version, flowcountid, st_x(StartPoint(geom)) AS lon,st_y(StartPoint(geom)) as lat, st_x(EndPoint(geom)) AS lon1,st_y(EndPoint(geom)) as lat1,type as type FROM flowcount_geometry limit %d,%d '
            % (int(start), int(limit)))

        # 通过fetchall方法获取全部查询结果
        result = cur.fetchall()
        # 关闭游标
        cur.close()
        # print result
        return result

    def findLineByDateModuleid(self, date,moduleid):
        # 创建游标
        cur = self.conn.cursor()

        # 对游标对象使用execute方法就可以执行普通的SQL语句
        # 做一次查询
        cur.execute(' SELECT id, moduleid, st_x(ST_STARTPOINT(geom)) AS lon,st_y(ST_STARTPOINT(geom)) as lat, st_x(ST_ENDPOINT(geom)) AS lon1,st_y(ST_ENDPOINT(geom)) as lat1 FROM '+self.table+' where DATE_FORMAT(startTime,\'%Y-%m-%d\')=\''+date+'\' and moduleid='+str(moduleid))

        # 通过fetchall方法获取全部查询结果
        result = cur.fetchall()
        # 关闭游标
        cur.close()
        # print result
        return result

    def findAllLineByDateModuleid(self, date,moduleid):
        # 创建游标
        cur = self.conn.cursor()

        # 对游标对象使用execute方法就可以执行普通的SQL语句
        # 做一次查询
        cur.execute(' SELECT id, moduleid, astext(geom) as geom FROM '+self.table+' where DATE_FORMAT(startTime,\'%Y-%m-%d\')=\''+date+'\' and moduleid='+str(moduleid))

        # 通过fetchall方法获取全部查询结果
        result = cur.fetchall()
        # 关闭游标
        cur.close()
        # print result
        return result

    def findPageByDate(self, start, limit, date):
        # 创建游标
        cur = self.conn.cursor()

        # 对游标对象使用execute方法就可以执行普通的SQL语句
        # 做一次查询
        cur.execute(
            ' SELECT id, moduleid,createtime,receivetime,packetnumber,powertimes,curflow,sumflow,version, flowcountid, st_x(ST_STARTPOINT(geom)) AS lon,st_y(ST_STARTPOINT(geom)) as lat, st_x(ST_ENDPOINT(geom)) AS lon1,st_y(ST_ENDPOINT(geom)) as lat1,type as type FROM flowcount_geometry where  DATE_FORMAT(createtime,\'%Y-%m-%d\')=%s order by id limit %d,%d '
            % (date, int(start), int(limit)))

        # 通过fetchall方法获取全部查询结果
        result = cur.fetchall()
        # 关闭游标
        cur.close()
        # print result
        return result

    def findWorkDate(self, month):
        # 创建游标
        cur = self.conn.cursor()

        # 对游标对象使用execute方法就可以执行普通的SQL语句
        # 做一次查询
        cur.execute(
            ' select DATE_FORMAT(createtime, \'%Y-%m-%d\') from flowcount_geometry where DATE_FORMAT(createtime, \'%Y-%m\') = %s  group BY DATE_FORMAT(createtime, \'%Y-%m-%d\') ' % (
            month))

        # 通过fetchall方法获取全部查询结果
        result = cur.fetchall()
        # 关闭游标
        cur.close()
        # print result
        return result

    def updateToGeometry(self):
        # 创建游标
        cur = self.conn.cursor()

        # 对游标对象使用execute方法就可以执行普通的SQL语句
        # 做一次查询
        cur.execute("    INSERT INTO flowcount.flowcount_geometry\n" +
                    "    (moduleid,createtime,receivetime, packetnumber, powertimes,curflow,sumflow,version,geom,flowcountid)" +
                    "    SELECT moduleid,createtime,receivetime, packetnumber, powertimes,curflow,sumflow,version," +
                    "    ST_GEOMETRYFROMTEXT(CONCAT('LINESTRING(',lat0,' ',lon0,',' ,lat1,' ',lon1,',',lat2,' ',lon2,',',lat3,' ',lon3,',',lat4,' ',lon4,',',lat5,' ',lon5,',',lat6,' ',lon6,',',lat7,' ',lon7,',',lat8,' ',lon8,',',lat9,' ',lon9,',',lat10,' ',lon10,',',lat11,' ',lon11,',',lat12,' ',lon12,',',lat13,' ',lon13,',',lat14,' ',lon14,')'))  as geom," +
                    "    id as flowcountid " +
                    "    FROM flowcount.flowcount_20176 " +
                    "    where id> (select IFNULL(max(flowcountid) ,0) from flowcount.flowcount_geometry where DATE_FORMAT(createtime, '%Y-%m') = '2017-06')  and (lat0>73 and  lon0>3 and lat1>73  and  lon1>3 and  lat2>73  and  lon2>3 and  lat3>73  and  lon3>3 and  lat4>73  and  lon4>3 and  lat5>73  and  lon5>3 and  lat6>73  and  lon6>3 and  lat7>73  and  lon7>3 and  lat8>73  and  lon8>3 and  lat9>73  and  lon9>3 and  lat10>73  and  lon10>3 and  lat11>73  and  lon11>3 and  lat12>73  and  lon12>3 and  lat13>73  and  lon13>3 and  lat14>73  and  lon14>3 and 136>lat0 and  54>lon0 and 136>lat1 and  54>lon1 and  136>lat2 and  54>lon2 and  136>lat3 and  54>lon3 and  136>lat4 and  54>lon4 and  136>lat5 and  54>lon5 and  136>lat6 and  54>lon6 and  136>lat7 and  54>lon7 and  136>lat8 and  54>lon8 and  136>lat9 and  54>lon9 and  136>lat10 and  54>lon10 and  136>lat11 and  54>lon11 and  136>lat12 and 54>lon12 and  136>lat13 and  54>lon13 and  136>lat14 and  54>lon14   ) ")

        # 关闭游标
        cur.close()


    def findByDateModuleid(self, date,moduleid):
        # 创建游标
        cur = self.conn.cursor()

        # 对游标对象使用execute方法就可以执行普通的SQL语句做一次查询
        cur.execute(' select  min(lon) as minlon,min(lat) as minlat,max(lon1) as maxlon,max(lat1) as maxlat from   (SELECT id, moduleid,startTime, st_x(ST_STARTPOINT(geom)) AS lon,st_y(ST_STARTPOINT(geom)) as lat, st_x(ST_ENDPOINT(geom)) AS lon1,st_y(ST_ENDPOINT(geom)) as lat1,type as type FROM '+self.table+' where  DATE_FORMAT(startTime,\'%Y-%m-%d\')=\''+date+'\' and moduleid='+str(moduleid)+') b' )

        # 通过fetchall方法获取全部查询结果
        result = cur.fetchone()
        # 关闭游标
        cur.close()
        # print result
        return result

    def findDateModuleidList(self):
        # 创建游标
        cur = self.conn.cursor()

        # 对游标对象使用execute方法就可以执行普通的SQL语句做一次查询
        cur.execute(' SELECT moduleid,DATE_FORMAT(startTime,\'%Y-%m-%d\')  as date FROM '+self.table+' group by DATE_FORMAT(startTime,\'%Y-%m-%d\'), moduleid' )

        # 通过fetchall方法获取全部查询结果
        result = cur.fetchall()
        # 关闭游标
        cur.close()
        # print result
        return result



    def findAllPolygonByDateModuleid(self, date,moduleid):
        # 创建游标
        cur = self.conn.cursor()

        # 对游标对象使用execute方法就可以执行普通的SQL语句
        # 做一次查询
        cur.execute(' SELECT id, moduleid, astext(geom) as geom FROM flytime_polygon_data where DATE_FORMAT(startTime,\'%Y-%m-%d\')=\''+date+'\' and moduleid='+str(moduleid))

        # 通过fetchall方法获取全部查询结果
        result = cur.fetchall()
        # 关闭游标
        cur.close()
        # print result
        return result

    def insertPolygon(self, moduleid, date,  geom):
        # 创建游标
        cur = self.conn.cursor()

        # 对游标对象使用execute方法就可以执行普通的SQL语句
        cur.execute("INSERT INTO flowcount.flytim_polygon_data(date,moduleid,geom, geom_gcj02, polygonArea,startTime) " +
                    "    VALUES (%s,%d,GeometryFromText(\'%s\'), GeometryFromText(\'%s\'), polygonArea,now())" % (date, int(moduleid), geom, geom,0) )

        # 关闭游标
        cur.close()
