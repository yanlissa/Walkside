from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.tiles.coordinates import lonlat_to_yandex_tms_tile


def find_cached_tile_path(base_dir, x, y, extensions):
    base_dir = Path(base_dir)

    for extension in extensions:
        path = base_dir / f"{x}_{y}{extension}"

        if path.exists():
            return path

    return None


def load_tiles_for_edge(edge_tiles, base_dir, extensions):
    tiles = []
    missing = []

    for tile in edge_tiles:
        x = tile["x"]
        y = tile["y"]
        path = find_cached_tile_path(base_dir, x, y, extensions)

        if path is None:
            missing.append({"z": tile["z"], "x": x, "y": y})
            continue

        tiles.append({
            "path": path,
            "x": x,
            "y": y,
            "image": Image.open(path).convert("RGB"),
        })

    return tiles, missing


def build_mosaic(tiles, tile_size=256):
    if not tiles:
        raise ValueError("Нет тайлов для построения мозаики")

    xs = [tile["x"] for tile in tiles]
    ys = [tile["y"] for tile in tiles]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    width = (max_x - min_x + 1) * tile_size
    height = (max_y - min_y + 1) * tile_size

    mosaic = Image.new("RGB", (width, height), color=(0, 0, 0))

    for tile in tiles:
        col = tile["x"] - min_x
        row = tile["y"] - min_y
        image = tile["image"].convert("RGB")

        if image.size != (tile_size, tile_size):
            image = image.resize((tile_size, tile_size), resample=Image.NEAREST)

        mosaic.paste(image, (col * tile_size, row * tile_size))

    mosaic_info = {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "width": width,
        "height": height,
        "tile_size": tile_size,
    }

    return mosaic, mosaic_info


def lonlat_to_pixel_on_mosaic(lat, lon, mosaic_info, z=20):
    x_float, y_float = lonlat_to_yandex_tms_tile(lat, lon, z)
    tile_size = mosaic_info["tile_size"]

    px = (x_float - mosaic_info["min_x"]) * tile_size
    py = (y_float - mosaic_info["min_y"]) * tile_size

    return px, py


def edge_to_pixels(edge_coords, mosaic_info, z=20):
    pixel_points = []

    for point in edge_coords:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue

        lon, lat = point[:2]

        pixel_points.append(
            lonlat_to_pixel_on_mosaic(
                lat=lat,
                lon=lon,
                mosaic_info=mosaic_info,
                z=z,
            )
        )

    return pixel_points


def overlay_mask_mosaic(image_mosaic, mask_mosaic, alpha=0.45, show_background=False):
    image_rgb = np.array(image_mosaic.convert("RGB"), dtype=np.float32)
    mask_rgb = np.array(mask_mosaic.convert("RGB"), dtype=np.float32)

    if show_background:
        mask_pixels = np.ones(mask_rgb.shape[:2], dtype=bool)
    else:
        mask_pixels = np.any(mask_rgb != np.array([0, 0, 0], dtype=np.float32), axis=-1)

    output = image_rgb.copy()
    output[mask_pixels] = (
        (1.0 - alpha) * image_rgb[mask_pixels]
        + alpha * mask_rgb[mask_pixels]
    )
    output = np.clip(output, 0, 255).astype(np.uint8)

    return Image.fromarray(output)


def point_inside_rect(px, py, width, height):
    return 0 <= px <= width and 0 <= py <= height


def segment_intersects_rect(p1, p2, width, height):
    x0, y0 = p1
    x1, y1 = p2

    if point_inside_rect(x0, y0, width, height):
        return True

    if point_inside_rect(x1, y1, width, height):
        return True

    left = 0
    right = width
    top = 0
    bottom = height

    dx = x1 - x0
    dy = y1 - y0

    p = [-dx, dx, -dy, dy]
    q = [x0 - left, right - x0, y0 - top, bottom - y0]

    u1 = 0.0
    u2 = 1.0

    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return False
        else:
            r = qi / pi

            if pi < 0:
                if r > u2:
                    return False
                if r > u1:
                    u1 = r
            else:
                if r < u1:
                    return False
                if r < u2:
                    u2 = r

    return True


def polyline_intersects_mosaic(pixel_points, mosaic_info):
    if len(pixel_points) < 2:
        return False

    width = mosaic_info["width"]
    height = mosaic_info["height"]

    for p1, p2 in zip(pixel_points[:-1], pixel_points[1:]):
        if segment_intersects_rect(p1, p2, width, height):
            return True

    return False


def polyline_inside_mosaic(pixel_points, mosaic_info):
    if len(pixel_points) < 2:
        return False

    width = mosaic_info["width"]
    height = mosaic_info["height"]

    for px, py in pixel_points:
        if not point_inside_rect(px, py, width, height):
            return False

    return True


def select_edges_for_mosaic(edges, mosaic_info, z=20, include_outside_edges=True):
    selected_edges = []

    for edge in edges:
        pixel_points = edge_to_pixels(
            edge_coords=edge["coords"],
            mosaic_info=mosaic_info,
            z=z,
        )

        if include_outside_edges:
            is_selected = polyline_intersects_mosaic(pixel_points, mosaic_info)
        else:
            is_selected = polyline_inside_mosaic(pixel_points, mosaic_info)

        if is_selected:
            selected_edge = edge.copy()
            selected_edge["pixel_points"] = pixel_points
            selected_edges.append(selected_edge)

    return selected_edges


def build_width_context_for_edge(
    edge,
    edge_tiles,
    images_dir,
    labels_dir,
    z=20,
    tile_size=256,
    mask_alpha=0.45,
    show_background=False,
    include_outside_edges=True,
):
    mask_tiles, missing_labels = load_tiles_for_edge(
        edge_tiles=edge_tiles,
        base_dir=labels_dir,
        extensions=(".png", ".jpg", ".jpeg"),
    )

    if missing_labels:
        return {
            "status": "missing_images_or_labels",
            "missing_images": [],
            "missing_labels": missing_labels,
        }

    mask_mosaic, mosaic_info = build_mosaic(
        mask_tiles,
        tile_size=tile_size,
    )

    selected_edges = select_edges_for_mosaic(
        edges=[edge],
        mosaic_info=mosaic_info,
        z=z,
        include_outside_edges=include_outside_edges,
    )

    return {
        "status": "ok",
        "image_mosaic": None,
        "mask_mosaic": mask_mosaic,
        "overlay": None,
        "mosaic_info": mosaic_info,
        "selected_edges": selected_edges,
        "missing_images": [],
        "missing_labels": [],
    }
