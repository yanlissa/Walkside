import cv2
import numpy as np
from shapely.geometry import Polygon


def polygon_parts(geom):
    if geom is None or geom.is_empty:
        return []

    if geom.geom_type == "Polygon":
        return [geom]

    if geom.geom_type == "MultiPolygon":
        return [part for part in geom.geoms if not part.is_empty]

    return []

def build_better_polygons_from_mask_image(
    mask_img,
    target_colors=((61, 61, 245), (255, 4, 4)),
    min_area=100,
    simplify_tolerance=4.0,
    smooth_radius=2,
    return_holes=False,
    min_hole_area=25,
):
    img_rgb = np.array(mask_img.convert("RGB"))

    combined_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)

    for color in target_colors:
        color = np.array(color, dtype=np.uint8)

        class_mask = cv2.inRange(
            img_rgb,
            color,
            color,
        )

        combined_mask = cv2.bitwise_or(
            combined_mask,
            class_mask,
        )

    if smooth_radius > 0:
        kernel = np.ones(
            (2 * smooth_radius + 1, 2 * smooth_radius + 1),
            np.uint8,
        )

        combined_mask = cv2.morphologyEx(
            combined_mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        combined_mask = cv2.morphologyEx(
            combined_mask,
            cv2.MORPH_OPEN,
            kernel,
        )

    contours, hierarchy = cv2.findContours(
        combined_mask,
        cv2.RETR_TREE if return_holes else cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    polygons = []
    holes = []

    for index, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        # Внутренние контуры сохраняем отдельно: они не должны влиять
        # на подбор полигона, поиск оси и итеративные разрезы.
        if return_holes and hierarchy is not None:
            parent = hierarchy[0, index, 3]
            if parent >= 0:
                # Только отверстия первого уровня. Вложенные островки
                # не являются внешними полигонами RETR_EXTERNAL.
                is_hole = hierarchy[0, parent, 3] == -1
                if is_hole and area >= min_hole_area:
                    points = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2)
                    if len(points) >= 3:
                        hole = Polygon(points)
                        if not hole.is_valid:
                            hole = hole.buffer(0)
                        holes.extend(polygon_parts(hole))
                continue

        if area < min_area:
            continue

        exterior = contour.reshape(-1, 2)

        if len(exterior) < 3:
            continue

        polygon = Polygon(exterior)

        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if polygon.is_empty:
            continue

        for part in polygon_parts(polygon):
            simplified = part.simplify(
                simplify_tolerance,
                preserve_topology=True,
            )

            if not simplified.is_valid:
                simplified = simplified.buffer(0)

            for simplified_part in polygon_parts(simplified):
                if simplified_part.area >= min_area:
                    polygons.append(simplified_part)

    polygons = sorted(
        polygons,
        key=lambda polygon: polygon.area,
        reverse=True,
    )

    if return_holes:
        return polygons, combined_mask, img_rgb, holes
    return polygons, combined_mask, img_rgb
