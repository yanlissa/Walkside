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

    contours, _ = cv2.findContours(
        combined_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    polygons = []

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
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

    return polygons, combined_mask, img_rgb