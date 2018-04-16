import unittest
from GeoUtil import GeoUtil

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_something(self):
        self.geoUtil = GeoUtil()

        lat = 40
        lon = 116
        gps = self.geoUtil.bd09_To_Gcj02(lat, lon)
        print "bd09_To_Gcj02 = " + str(gps["lat"]) + "," + str(gps["lon"])

        gps = self.geoUtil.gcj_To_Gps84(lat, lon)
        print "gcj_To_Gps84 = " + str(gps["lat"]) + "," + str(gps["lon"])

        gps = self.geoUtil.gps84_To_Gcj02(lat, lon)
        print "gps84_To_Gcj02 = " + str(gps["lat"]) + "," + str(gps["lon"])

        gps = self.geoUtil.bd09_To_Gcj02(lat, lon)
        print "bd09_To_Gcj02 = " + str(gps["lat"]) + "," + str(gps["lon"])

        gps = self.geoUtil.gcj02_To_Bd09(lat, lon)
        print "gcj02_To_Bd09 = " + str(gps["lat"]) + "," + str(gps["lon"])

        gps = self.geoUtil.transform(lat, lon)
        print "transform = " + str(gps["lat"]) + "," + str(gps["lon"])

        xy = self.geoUtil.lonLat2Mercator(lon, lat)
        print "lonLat2Mercator = " + str(xy[0]) + "," + str(xy[1])

        xy = self.geoUtil.Mercator2lonLat(12913060.932019578, 4865942.280741853)
        print "Mercator2lonLat = " + str(xy[0]) + "," + str(xy[1])


        # bd09_To_Gcj02 = 39.99365221474683,115.99362031284348
        # gcj_To_Gps84 = 39.99878627294799,115.99385881033336
        # gps84_To_Gcj02 = 40.00121372705201,116.00614118966664
        # bd09_To_Gcj02 = 39.99365221474683,115.99362031284348
        # gcj02_To_Bd09 = 40.006347999819994,116.006379999478
        # transform = 40.00121372705201,116.00614118966664
        # lonLat2Mercator = 1.2913060932019578E7,4865942.280741853
        # Mercator2lonLat = 116.00000001614593,40.00000001318511

if __name__ == '__main__':
    unittest.main()
