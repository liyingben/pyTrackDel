from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import skimage.io as io
from skimage import measure, data, color
from shapely.geometry import Polygon

from Util import Util


from shapely.wkt import dumps, loads
from shapely.geometry import MultiLineString
from shapely.validation import explain_validity
from PIL import Image
import ImageFilter as ImFilter
import ImageChops as ImChops
import ImageFont as ImFont
import ImageDraw as ImDraw
import time


pixScale = Util.pixScale

def createTileBox(list,startLon, startLat, bounds):
    newImg = Image.new("RGBA", (256, 256), (255, 255, 255))
    draw = ImDraw.Draw(newImg)
    for geo in list:

        if geo.geom_type == 'LineString':
            line = geoToPix(startLon, startLat, bounds,geo)
            draw.line(line, (0, 0, 0),width=2)

        if geo.geom_type == 'Polygon':
            polygon = geoToPix(startLon, startLat, bounds,geo)
            draw.polygon(polygon, fill=(0, 0, 0), outline=(0, 0, 0))
    newImg.save("/tmp/newImg.png")



def geoToPix(startLon, startLat, bounds,geo):
    list=[]
    if geo.geom_type == 'LineString':
        list = geo.coords
    if geo.geom_type == 'Polygon':
        list = geo.exterior.coords

    coords=[]
    for xy in list:
        px = int(((xy[0] - bounds[0]) * pixScale))
        py = 256-int(((xy[1] - bounds[1]) * pixScale))
        coords.append((px,py))

    return coords



def main():

    start = time.time()

    # db = DataBase(host='localhost',
    #               port=3306,
    #               user='root',
    #               passwd='root',
    #               db='flowcount',
    #               table='flowcount_geometry_201710')

    # list = db.findDateModuleidList()
    lineList = []
    # for wkt in list:
    #     lineList.append(loads(wkt[2]))
    lineList.append(loads('LineString(128.632192 45.401568,128.632252 45.401539,128.632312 45.401511,128.632372 45.401484,128.632432 45.401458,128.632494 45.401431,128.632556 45.401405,128.63262 45.401381,128.632684 45.401356,128.632747 45.401332,128.632811 45.401307,128.632857 45.401289,128.632918 45.401265,128.632979 45.401241,128.633042 45.401219)'))
    lines = MultiLineString(lineList)

    print(lines.bounds)
    # newImg = Image.new("RGBA", (256, 256), (255, 255, 255))
    # newImg.paste(img,(2,2))
    # draw = ImDraw.Draw(newImg)
    # draw.arc((0, 0, 202, 202), 0, 135, (0, 255, 0))
    # draw.arc((0, 0, 205, 205), 0, 135, (255, 0, 0))
    # draw.arc((0, 0, 208, 208), 0, 135, (0, 0, 255))
    # draw.arc((0, 0, 211, 211), 0, 135, (255, 255, 0))
    # draw.arc((0, 0, 212, 212), 0, 135, (255, 0, 255))
    # # Cavon2 = Im.new('RGB',(200,300),(255,255,255))
    # draw.ellipse((0, 0, 30, 40), (0, 255, 0))
    # draw.ellipse((20, 20, 40, 30), (255, 125, 30))
    # draw.line(((60, 60), (90, 60), (90, 90), (60, 90), (60, 60)), (255, 0, 0))
    # draw.point((100, 100), (255, 0, 255))
    # draw.polygon([(60.5, 60.4), (90, 60), (90, 90), (60, 90)], fill="red", outline="green")
    # newImg.save("/tmp/newImg.png")
    startLon = lines.bounds[0]
    startLat = lines.bounds[1]
    createTileBox(lines, startLon, startLat, (lines.bounds[0],lines.bounds[1],lines.bounds[0]+pixScale*256.0,lines.bounds[1]+pixScale*256.0))


main()
