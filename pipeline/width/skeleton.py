import math

import cv2
import numpy as np
from shapely import affinity
from shapely.geometry import LineString
from shapely.ops import nearest_points
from skimage.morphology import skeletonize

from pipeline.width.polygons import polygon_parts


def get_representative_line(line_geom):
    if line_geom is None or line_geom.is_empty:
        return None
    if line_geom.geom_type == 'LineString':
        return line_geom
    if line_geom.geom_type == 'MultiLineString':
        lines = list(line_geom.geoms)
        if not lines:
            return None
        return max(lines, key=lambda geom: geom.length)
    return None


def get_direction_at_distance(line, distance, delta=3.0):
    d1 = max(0, distance - delta)
    d2 = min(line.length, distance + delta)
    p1 = line.interpolate(d1)
    p2 = line.interpolate(d2)
    vx = p2.x - p1.x
    vy = p2.y - p1.y
    norm = math.hypot(vx, vy)
    if norm == 0:
        return None
    return (vx / norm, vy / norm)


def normalize_vector(vx, vy):
    norm = math.hypot(vx, vy)
    if norm == 0:
        return None
    return (vx / norm, vy / norm)


def polygon_to_local_mask(polygon, pad=5):
    if polygon is None or polygon.is_empty:
        return (None, None)
    polygon = max(polygon_parts(polygon), key=lambda p: p.area)
    minx, miny, maxx, maxy = polygon.bounds
    minx_i = int(math.floor(minx)) - pad
    miny_i = int(math.floor(miny)) - pad
    maxx_i = int(math.ceil(maxx)) + pad
    maxy_i = int(math.ceil(maxy)) + pad
    width = maxx_i - minx_i + 1
    height = maxy_i - miny_i + 1
    if width <= 0 or height <= 0:
        return (None, None)
    mask = np.zeros((height, width), dtype=np.uint8)
    exterior = np.array([[int(round(x - minx_i)), int(round(y - miny_i))] for x, y in polygon.exterior.coords], dtype=np.int32)
    cv2.fillPoly(mask, [exterior], 255)
    for interior in polygon.interiors:
        hole = np.array([[int(round(x - minx_i)), int(round(y - miny_i))] for x, y in interior.coords], dtype=np.int32)
        cv2.fillPoly(mask, [hole], 0)
    return (mask, (minx_i, miny_i))


def get_skeleton_neighbors(pixel, skeleton_pixels):
    r, c = pixel
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    neighbors = []
    for dr, dc in shifts:
        candidate = (r + dr, c + dc)
        if candidate in skeleton_pixels:
            neighbors.append(candidate)
    return neighbors


def pixel_to_xy(pixel, offset):
    r, c = pixel
    x0, y0 = offset
    return (x0 + c, y0 + r)


def trace_skeleton_polyline_from_node(start_pixel, next_pixel, skeleton_pixels, node_pixels, visited_edges):
    path = [start_pixel, next_pixel]
    previous = start_pixel
    current = next_pixel
    visited_edges.add(frozenset([start_pixel, next_pixel]))
    while True:
        if current in node_pixels and current != start_pixel:
            break
        neighbors = get_skeleton_neighbors(current, skeleton_pixels)
        next_candidates = [item for item in neighbors if item != previous]
        if not next_candidates:
            break
        next_candidate = None
        for candidate in next_candidates:
            edge_key = frozenset([current, candidate])
            if edge_key not in visited_edges:
                next_candidate = candidate
                break
        if next_candidate is None:
            break
        visited_edges.add(frozenset([current, next_candidate]))
        previous = current
        current = next_candidate
        path.append(current)
    return path


def trace_cycle_skeleton(skeleton_pixels):
    if not skeleton_pixels:
        return []
    start = next(iter(skeleton_pixels))
    path = [start]
    previous = None
    current = start
    visited = set()
    while True:
        candidates = [item for item in get_skeleton_neighbors(current, skeleton_pixels) if item != previous]
        if not candidates:
            break
        next_pixel = None
        for candidate in candidates:
            edge_key = frozenset([current, candidate])
            if edge_key not in visited:
                next_pixel = candidate
                break
        if next_pixel is None:
            break
        visited.add(frozenset([current, next_pixel]))
        previous = current
        current = next_pixel
        if current == start:
            break
        path.append(current)
        if len(path) > len(skeleton_pixels) + 10:
            break
    return path


def build_polygon_skeleton_lines(polygon, pad=5, min_line_length=3.0):
    mask, offset = polygon_to_local_mask(polygon=polygon, pad=pad)
    if mask is None:
        return []
    skeleton = skeletonize(mask > 0)
    rows, cols = np.where(skeleton)
    skeleton_pixels = set(zip(rows.tolist(), cols.tolist()))
    if not skeleton_pixels:
        return []
    degree = {pixel: len(get_skeleton_neighbors(pixel, skeleton_pixels)) for pixel in skeleton_pixels}
    node_pixels = {pixel for pixel, deg in degree.items() if deg != 2}
    skeleton_lines = []
    visited_edges = set()
    if node_pixels:
        for node in node_pixels:
            for neighbor in get_skeleton_neighbors(node, skeleton_pixels):
                edge_key = frozenset([node, neighbor])
                if edge_key in visited_edges:
                    continue
                path = trace_skeleton_polyline_from_node(start_pixel=node, next_pixel=neighbor, skeleton_pixels=skeleton_pixels, node_pixels=node_pixels, visited_edges=visited_edges)
                if len(path) < 2:
                    continue
                coords = [pixel_to_xy(pixel, offset) for pixel in path]
                line = LineString(coords)
                if line.length >= min_line_length:
                    skeleton_lines.append(line)
    else:
        path = trace_cycle_skeleton(skeleton_pixels)
        if len(path) >= 2:
            coords = [pixel_to_xy(pixel, offset) for pixel in path]
            line = LineString(coords)
            if line.length >= min_line_length:
                skeleton_lines.append(line)
    return skeleton_lines


def approximate_axis_lines(skeleton_lines, simplify_tolerance=8.0, min_segment_length=8.0):
    axis_segments = []
    for line in skeleton_lines:
        if line is None or line.is_empty:
            continue
        simplified = line.simplify(simplify_tolerance, preserve_topology=False)
        if simplified.geom_type != 'LineString':
            continue
        coords = list(simplified.coords)
        for p1, p2 in zip(coords[:-1], coords[1:]):
            segment = LineString([p1, p2])
            if segment.length >= min_segment_length:
                axis_segments.append(segment)
    return axis_segments


def segment_direction(segment):
    if segment is None or segment.is_empty:
        return None
    coords = list(segment.coords)
    if len(coords) < 2:
        return None
    x1, y1 = coords[0]
    x2, y2 = coords[-1]
    return normalize_vector(x2 - x1, y2 - y1)


def densify_line(line, step=10):
    line = get_representative_line(line)
    if line is None or line.is_empty or line.length == 0:
        return None
    distances = np.arange(0, line.length + 1e-06, step)
    points = []
    for distance in distances:
        point = line.interpolate(float(distance))
        points.append((point.x, point.y))
    end = line.interpolate(line.length)
    end_point = (end.x, end.y)
    if not points or points[-1] != end_point:
        points.append(end_point)
    if len(points) < 2:
        return None
    return LineString(points)


def split_line_to_segments(line):
    line = get_representative_line(line)
    if line is None or line.is_empty:
        return []
    coords = list(line.coords)
    segments = []
    for p1, p2 in zip(coords[:-1], coords[1:]):
        segment = LineString([p1, p2])
        if segment.length > 0:
            segments.append(segment)
    return segments


def evaluate_translation_against_axis(edge_line, axis_segments, dx, dy, edge_step=10, max_match_distance=120, min_parallel_cos=0.75):
    translated_edge = translate_edge_line(edge_line=edge_line, dx=dx, dy=dy)
    if translated_edge is None or translated_edge.is_empty:
        return None
    dense_edge = densify_line(line=translated_edge, step=edge_step)
    if dense_edge is None:
        return None
    edge_segments = split_line_to_segments(dense_edge)
    if not edge_segments:
        return None
    matches = []
    for edge_segment in edge_segments:
        edge_direction = segment_direction(edge_segment)
        if edge_direction is None:
            continue
        ex, ey = edge_direction
        edge_midpoint = edge_segment.interpolate(0.5, normalized=True)
        best_match = None
        best_score = -1e+18
        for axis_segment in axis_segments:
            axis_direction = segment_direction(axis_segment)
            if axis_direction is None:
                continue
            ax, ay = axis_direction
            dot = abs(ex * ax + ey * ay)
            if dot < min_parallel_cos:
                continue
            distance = edge_midpoint.distance(axis_segment)
            if distance > max_match_distance:
                continue
            _, axis_point = nearest_points(edge_midpoint, axis_segment)
            score = dot * 10.0 - distance / max(max_match_distance, 1)
            if score > best_score:
                best_score = score
                best_match = {'edge_segment': edge_segment, 'axis_segment': axis_segment, 'edge_point': edge_midpoint, 'axis_point': axis_point, 'edge_direction': edge_direction, 'axis_direction': axis_direction, 'dot': dot, 'distance': distance, 'score': score}
        if best_match is not None:
            matches.append(best_match)
    coverage_ratio = len(matches) / len(edge_segments)
    if not matches:
        return {'translated_edge': translated_edge, 'edge_segments': edge_segments, 'matches': [], 'coverage_ratio': coverage_ratio, 'mean_dot': 0.0, 'mean_distance': None}
    mean_dot = float(np.mean([m['dot'] for m in matches]))
    mean_distance = float(np.mean([m['distance'] for m in matches]))
    return {'translated_edge': translated_edge, 'edge_segments': edge_segments, 'matches': matches, 'coverage_ratio': float(coverage_ratio), 'mean_dot': mean_dot, 'mean_distance': mean_distance}


def estimate_best_global_translation_from_axis(edge_line, axis_segments, edge_step=10, max_candidate_distance=220, max_match_distance=120, min_parallel_cos=0.75, shift_cluster_radius=12, max_candidates=250, shift_length_weight=0.001):
    dense_edge = densify_line(line=edge_line, step=edge_step)
    if dense_edge is None:
        return None
    edge_segments = split_line_to_segments(dense_edge)
    if not edge_segments:
        return None
    clusters = {}
    for edge_segment in edge_segments:
        edge_direction = segment_direction(edge_segment)
        if edge_direction is None:
            continue
        ex, ey = edge_direction
        edge_midpoint = edge_segment.interpolate(0.5, normalized=True)
        for axis_segment in axis_segments:
            axis_direction = segment_direction(axis_segment)
            if axis_direction is None:
                continue
            ax, ay = axis_direction
            dot = abs(ex * ax + ey * ay)
            if dot < min_parallel_cos:
                continue
            distance = edge_midpoint.distance(axis_segment)
            if distance > max_candidate_distance:
                continue
            _, axis_point = nearest_points(edge_midpoint, axis_segment)
            dx = axis_point.x - edge_midpoint.x
            dy = axis_point.y - edge_midpoint.y
            key = (int(round(dx / shift_cluster_radius)), int(round(dy / shift_cluster_radius)))
            if key not in clusters:
                clusters[key] = {'dxs': [], 'dys': [], 'dots': [], 'distances': []}
            clusters[key]['dxs'].append(dx)
            clusters[key]['dys'].append(dy)
            clusters[key]['dots'].append(dot)
            clusters[key]['distances'].append(distance)
    if not clusters:
        return None
    candidates = []
    for cluster in clusters.values():
        dxs = np.array(cluster['dxs'], dtype=float)
        dys = np.array(cluster['dys'], dtype=float)
        dots = np.array(cluster['dots'], dtype=float)
        distances = np.array(cluster['distances'], dtype=float)
        dx = float(np.median(dxs))
        dy = float(np.median(dys))
        residuals = np.sqrt((dxs - dx) ** 2 + (dys - dy) ** 2)
        shift_std = float(np.median(residuals))
        shift_length = float(math.hypot(dx, dy))
        seed_score = len(dxs) * 2.0 + float(np.mean(dots)) * 5.0 - float(np.mean(distances)) / max(max_candidate_distance, 1) - shift_length_weight * shift_length - 0.05 * shift_std / max(edge_step, 1)
        candidates.append({'dx': dx, 'dy': dy, 'shift_length': shift_length, 'shift_std': shift_std, 'seed_score': seed_score, 'cluster_count': len(dxs)})
    candidates = sorted(candidates, key=lambda item: item['seed_score'], reverse=True)[:max_candidates]
    best = None
    best_score = -1e+18
    for candidate in candidates:
        evaluation = evaluate_translation_against_axis(edge_line=edge_line, axis_segments=axis_segments, dx=candidate['dx'], dy=candidate['dy'], edge_step=edge_step, max_match_distance=max_match_distance, min_parallel_cos=min_parallel_cos)
        if evaluation is None:
            continue
        mean_distance = evaluation['mean_distance']
        if mean_distance is None:
            mean_distance_norm = 1.0
        else:
            mean_distance_norm = mean_distance / max(max_match_distance, 1)
        score = 2.0 * evaluation['coverage_ratio'] + 0.8 * evaluation['mean_dot'] - 0.6 * mean_distance_norm - shift_length_weight * candidate['shift_length'] - 0.1 * candidate['shift_std'] / max(edge_step, 1)
        item = {**candidate, **evaluation, 'global_score': float(score)}
        if score > best_score:
            best_score = score
            best = item
    return best


def translate_edge_line(edge_line, dx, dy):
    edge_line = get_representative_line(edge_line)
    if edge_line is None or edge_line.is_empty:
        return None
    return affinity.translate(edge_line, xoff=dx, yoff=dy)


def score_edge_against_polygon_axis_by_dot(edge_line, polygon, skeleton_pad=5, skeleton_min_line_length=3.0, axis_simplify_tolerance=8.0, axis_min_segment_length=8.0, edge_step=10, max_match_distance=120, min_parallel_cos=0.75, distance_sigma=120.0):
    skeleton_lines = build_polygon_skeleton_lines(polygon=polygon, pad=skeleton_pad, min_line_length=skeleton_min_line_length)
    axis_segments = approximate_axis_lines(skeleton_lines=skeleton_lines, simplify_tolerance=axis_simplify_tolerance, min_segment_length=axis_min_segment_length)
    if not axis_segments:
        axis_segments = skeleton_lines
    if not axis_segments:
        return {'axis_status': 'no_axis_segments', 'axis_score': 0.0, 'mean_dot': 0.0, 'coverage_ratio': 0.0, 'mean_distance': None, 'shift_dx': None, 'shift_dy': None, 'shift_length': None, 'shift_std': None, 'translated_edge': None, 'skeleton_lines': skeleton_lines, 'axis_segments': axis_segments, 'edge_segments': [], 'matches': []}
    translation = estimate_best_global_translation_from_axis(edge_line=edge_line, axis_segments=axis_segments, edge_step=edge_step, max_candidate_distance=max_match_distance * 2.0, max_match_distance=max_match_distance, min_parallel_cos=min_parallel_cos, shift_cluster_radius=12, max_candidates=250, shift_length_weight=0.001)
    if translation is None:
        dense_edge = densify_line(line=edge_line, step=edge_step)
        edge_segments = [] if dense_edge is None else split_line_to_segments(dense_edge)
        return {'axis_status': 'no_global_translation', 'axis_score': 0.0, 'mean_dot': 0.0, 'coverage_ratio': 0.0, 'mean_distance': None, 'shift_dx': None, 'shift_dy': None, 'shift_length': None, 'shift_std': None, 'translated_edge': None, 'skeleton_lines': skeleton_lines, 'axis_segments': axis_segments, 'edge_segments': edge_segments, 'matches': []}
    mean_distance = translation['mean_distance']
    if mean_distance is None:
        distance_score = 0.0
    else:
        distance_score = math.exp(-mean_distance / distance_sigma)
    shift_consistency = math.exp(-translation['shift_std'] / max(edge_step, 1))
    axis_score = (0.6 * translation['coverage_ratio'] + 0.25 * translation['mean_dot'] + 0.15 * shift_consistency) * distance_score
    return {'axis_status': 'ok_global_translation', 'axis_score': float(axis_score), 'mean_dot': translation['mean_dot'], 'coverage_ratio': float(translation['coverage_ratio']), 'mean_distance': translation['mean_distance'], 'shift_dx': translation['dx'], 'shift_dy': translation['dy'], 'shift_length': translation['shift_length'], 'shift_std': translation['shift_std'], 'translated_edge': translation['translated_edge'], 'skeleton_lines': skeleton_lines, 'axis_segments': axis_segments, 'edge_segments': translation['edge_segments'], 'matches': translation['matches'], 'global_score': translation['global_score'], 'cluster_count': translation['cluster_count']}
