import math

from pipeline.tiles.coordinates import lonlat_to_yandex_tms_tile


def is_position(value):
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    )


def edge_to_bbox(edge_coords, buffer_degrees=0.0):
    points = [point for point in edge_coords if is_position(point)]

    if not points:
        raise ValueError("У ребра нет корректных координат")

    lons = [point[0] for point in points]
    lats = [point[1] for point in points]

    return (
        min(lons) - buffer_degrees,
        min(lats) - buffer_degrees,
        max(lons) + buffer_degrees,
        max(lats) + buffer_degrees,
    )


def get_tiles_for_single_edge(
    edge,
    z,
    context_px,
    tile_size=256,
    buffer_degrees=0.0,
):
    bbox = edge_to_bbox(edge["coords"], buffer_degrees=buffer_degrees)
    min_lon, min_lat, max_lon, max_lat = bbox

    points = [
        lonlat_to_yandex_tms_tile(min_lat, min_lon, z),
        lonlat_to_yandex_tms_tile(min_lat, max_lon, z),
        lonlat_to_yandex_tms_tile(max_lat, min_lon, z),
        lonlat_to_yandex_tms_tile(max_lat, max_lon, z),
    ]

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    context_tiles = context_px / tile_size
    n = 2 ** z

    x_min = max(0, math.floor(min(xs) - context_tiles))
    x_max = min(n - 1, math.floor(max(xs) + context_tiles))

    y_min = max(0, math.floor(min(ys) - context_tiles))
    y_max = min(n - 1, math.floor(max(ys) + context_tiles))

    tiles = []

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tiles.append({
                "z": z,
                "x": x,
                "y": y,
            })

    return tiles


def build_edge_tile_index(
    edges,
    z,
    context_px,
    tile_size=256,
    buffer_degrees=0.0,
    progress_callback=None,
):
    edge_tile_index = {}
    unique_tiles = {}
    total_edges = len(edges)

    if progress_callback is not None:
        progress_callback(0, total_edges)

    for edge_index, edge in enumerate(edges):
        edge_tiles = get_tiles_for_single_edge(
            edge=edge,
            z=z,
            context_px=context_px,
            tile_size=tile_size,
            buffer_degrees=buffer_degrees,
        )

        edge_tile_index[edge_index] = edge_tiles

        for tile in edge_tiles:
            key = (tile["z"], tile["x"], tile["y"])

            if key not in unique_tiles:
                unique_tiles[key] = {
                    "z": tile["z"],
                    "x": tile["x"],
                    "y": tile["y"],
                    "edge_indices": [],
                }

            unique_tiles[key]["edge_indices"].append(edge_index)

        if progress_callback is not None:
            progress_callback(edge_index + 1, total_edges)

    return edge_tile_index, unique_tiles


def tile_key(tile):
    return (int(tile["z"]), int(tile["x"]), int(tile["y"]))


def filter_edges_with_complete_tiles(
    edges,
    edge_tile_index,
    missing_tiles,
):
    """
    Исключает из рабочего набора каждое ребро, для которого отсутствует
    хотя бы один обязательный тайл. Исходные объекты рёбер не изменяются.

    Возвращаемый `edge_tile_index` перенумерован относительно списка
    `eligible_edges`. В самих рёбрах сохраняется `feature_idx`, поэтому
    рассчитанная ширина по-прежнему записывается в правильный объект GeoJSON.
    """
    missing_keys = {tile_key(tile) for tile in missing_tiles}

    eligible_edges = []
    eligible_edge_tile_index = {}
    eligible_unique_tiles = {}
    skipped_edges = []

    for original_edge_index, edge in enumerate(edges):
        required_tiles = list(edge_tile_index.get(original_edge_index, []))
        edge_missing_tiles = [
            tile for tile in required_tiles
            if tile_key(tile) in missing_keys
        ]

        if edge_missing_tiles:
            skipped_edges.append({
                "original_edge_index": original_edge_index,
                "edge": edge,
                "missing_tiles": edge_missing_tiles,
            })
            continue

        new_edge_index = len(eligible_edges)
        eligible_edges.append(edge)
        eligible_edge_tile_index[new_edge_index] = required_tiles

        for tile in required_tiles:
            key = tile_key(tile)

            if key not in eligible_unique_tiles:
                eligible_unique_tiles[key] = {
                    "z": tile["z"],
                    "x": tile["x"],
                    "y": tile["y"],
                    "edge_indices": [],
                }

            eligible_unique_tiles[key]["edge_indices"].append(new_edge_index)

    return {
        "eligible_edges": eligible_edges,
        "edge_tile_index": eligible_edge_tile_index,
        "unique_tiles": eligible_unique_tiles,
        "skipped_edges": skipped_edges,
        "missing_tile_keys": missing_keys,
    }
