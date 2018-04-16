# coding=utf-8


from shapely.geometry import Point
from shapely.ops import cascaded_union
polygons = [Point(i, 0).buffer(0.7) for i in range(5)]
cascaded_union(polygons)