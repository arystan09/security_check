import cv2
import numpy as np
from typing import List, Tuple


def point_in_polygon(point: Tuple[int, int], polygon: List[List[int]]) -> bool:
    points_array = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(points_array, point, False)
    return result >= 0


def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)


def check_person_in_zones(center: Tuple[int, int], zones: List[dict]) -> List[int]:
    intersecting_zones = []
    for zone in zones:
        if point_in_polygon(center, zone['points']):
            intersecting_zones.append(zone['id'])
    return intersecting_zones


