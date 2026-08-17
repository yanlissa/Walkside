import math


def lonlat_to_yandex_tms_tile(lat, lon, z=20):
    n = 2 ** z

    lat = max(min(lat, 85.05112878), -85.05112878)
    lat_rad = math.radians(lat)

    x = (lon + 180.0) / 360.0 * n

    y = (
        1.0
        - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi
    ) / 2.0 * n

    return x, y