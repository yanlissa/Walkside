import math

import numpy as np
from shapely.geometry import LineString
from shapely.ops import split

from pipeline.width.measurement import (
    measure_width_along_edge_with_processed_polygon,
    width_quality,
)
from pipeline.width.polygons import polygon_parts
from pipeline.width.skeleton import get_direction_at_distance, get_representative_line


def signed_polygon_area(coords):
    area = 0.0
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def find_concave_vertices(polygon, angle_threshold=0.0):
    if polygon is None or polygon.is_empty:
        return []
    if polygon.geom_type != 'Polygon':
        return []
    coords = list(polygon.exterior.coords)
    if len(coords) < 4:
        return []
    orientation = 1 if signed_polygon_area(coords) > 0 else -1
    concave_points = []
    for i in range(1, len(coords) - 1):
        p_prev = coords[i - 1]
        p = coords[i]
        p_next = coords[i + 1]
        ax = p[0] - p_prev[0]
        ay = p[1] - p_prev[1]
        bx = p_next[0] - p[0]
        by = p_next[1] - p[1]
        cross = ax * by - ay * bx
        if orientation * cross < -angle_threshold:
            concave_points.append(p)
    return concave_points


def get_unit_vector_between_points(point_a, point_b):
    x1, y1 = point_a
    x2, y2 = point_b
    vx = x2 - x1
    vy = y2 - y1
    norm = math.hypot(vx, vy)
    if norm == 0:
        return None
    return (vx / norm, vy / norm)


def extend_segment_between_points(point_a, point_b, extension):
    x1, y1 = point_a
    x2, y2 = point_b
    vx = x2 - x1
    vy = y2 - y1
    norm = math.hypot(vx, vy)
    if norm == 0:
        return LineString([point_a, point_b])
    vx /= norm
    vy /= norm
    return LineString([(x1 - vx * extension, y1 - vy * extension), (x2 + vx * extension, y2 + vy * extension)])


def split_polygon_by_line(polygon, cut_line, min_area):
    try:
        result = split(polygon, cut_line)
    except Exception:
        return [polygon]
    pieces = []
    for geom in result.geoms:
        if not geom.is_empty and geom.area >= min_area:
            pieces.append(geom)
    return pieces


def split_and_keep_edge_piece(polygon, split_line, edge_line, edge_buffer, min_area_after_split, min_removed_area, max_removed_area_ratio):
    pieces = split_polygon_by_line(polygon=polygon, cut_line=split_line, min_area=min_area_after_split)
    if len(pieces) <= 1:
        return None
    edge_area = edge_line.buffer(edge_buffer)
    kept_piece = max(pieces, key=lambda piece: piece.intersection(edge_area).area)
    removed_pieces = [piece for piece in pieces if piece is not kept_piece]
    removed_area = sum((piece.area for piece in removed_pieces))
    total_area = sum((piece.area for piece in pieces))
    if removed_area < min_removed_area:
        return None
    if removed_area / max(total_area, 1e-06) > max_removed_area_ratio:
        return None
    return (kept_piece, removed_pieces)


def find_best_local_cut(polygon, edge_line, step, max_width, simplify_tolerance, angle_threshold, min_parallel_cos, split_extension, edge_buffer, min_area_after_split, min_removed_area, max_removed_area_ratio, local_direction_delta, max_cut_distance_to_edge, min_valid_ratio):
    simplified = polygon.simplify(tolerance=simplify_tolerance, preserve_topology=True)
    if not simplified.is_valid:
        simplified = simplified.buffer(0)
    best = None
    for simplified_part in polygon_parts(simplified):
        concave_points = find_concave_vertices(polygon=simplified_part, angle_threshold=angle_threshold)
        for i in range(len(concave_points)):
            for j in range(i + 1, len(concave_points)):
                point_a = concave_points[i]
                point_b = concave_points[j]
                pair_direction = get_unit_vector_between_points(point_a, point_b)
                if pair_direction is None:
                    continue
                visual_cut_line = LineString([point_a, point_b])
                cut_midpoint = visual_cut_line.interpolate(0.5, normalized=True)
                distance_along_edge = edge_line.project(cut_midpoint)
                projected_point = edge_line.interpolate(distance_along_edge)
                if cut_midpoint.distance(projected_point) > max_cut_distance_to_edge:
                    continue
                local_direction = get_direction_at_distance(line=edge_line, distance=distance_along_edge, delta=local_direction_delta)
                if local_direction is None:
                    continue
                ex, ey = local_direction
                px, py = pair_direction
                parallel_cos = abs(ex * px + ey * py)
                if parallel_cos < min_parallel_cos:
                    continue
                split_line = extend_segment_between_points(point_a=point_a, point_b=point_b, extension=split_extension)
                split_result = split_and_keep_edge_piece(polygon=polygon, split_line=split_line, edge_line=edge_line, edge_buffer=edge_buffer, min_area_after_split=min_area_after_split, min_removed_area=min_removed_area, max_removed_area_ratio=max_removed_area_ratio)
                if split_result is None:
                    continue
                candidate_polygon, removed_pieces = split_result
                measurements = measure_width_along_edge_with_processed_polygon(edge_line=edge_line, processed_polygon=candidate_polygon, step=step, max_width=max_width)
                quality = width_quality(measurements)
                if quality['valid_ratio'] < min_valid_ratio:
                    continue
                candidate_score = quality['score'] + 0.05 * parallel_cos
                candidate = {'score': float(candidate_score), 'processed_polygon': candidate_polygon, 'removed_pieces': removed_pieces, 'visual_cut_line': visual_cut_line, 'split_line': split_line, 'measurements': measurements, 'quality': quality, 'parallel_cos': float(parallel_cos), 'distance_along_edge': float(distance_along_edge)}
                if best is None or candidate['score'] > best['score']:
                    best = candidate
    return best


def iterative_local_cuts(polygon, edge_line, step=10, max_width=120, max_cuts=5, simplify_tolerance=20.0, angle_threshold=0.0, min_parallel_cos=0.75, split_extension=5, edge_buffer=5, min_area_after_split=20, min_removed_area=50, max_removed_area_ratio=0.75, local_direction_delta=25.0, max_cut_distance_to_edge=120.0, min_valid_ratio=0.5, min_rel_mae_improvement=0.03, min_valid_ratio_keep=0.8):
    edge_line = get_representative_line(edge_line)
    current_polygon = polygon
    accepted_cuts = []
    removed_pieces_all = []
    current_measurements = measure_width_along_edge_with_processed_polygon(edge_line=edge_line, processed_polygon=current_polygon, step=step, max_width=max_width)
    current_quality = width_quality(current_measurements)
    for _ in range(max_cuts):
        best_cut = find_best_local_cut(polygon=current_polygon, edge_line=edge_line, step=step, max_width=max_width, simplify_tolerance=simplify_tolerance, angle_threshold=angle_threshold, min_parallel_cos=min_parallel_cos, split_extension=split_extension, edge_buffer=edge_buffer, min_area_after_split=min_area_after_split, min_removed_area=min_removed_area, max_removed_area_ratio=max_removed_area_ratio, local_direction_delta=local_direction_delta, max_cut_distance_to_edge=max_cut_distance_to_edge, min_valid_ratio=min_valid_ratio)
        if best_cut is None:
            break
        old_rel_mae = current_quality['rel_mae']
        new_rel_mae = best_cut['quality']['rel_mae']
        old_valid_ratio = current_quality['valid_ratio']
        new_valid_ratio = best_cut['quality']['valid_ratio']
        improvement_ok = new_rel_mae < old_rel_mae - min_rel_mae_improvement
        coverage_ok = new_valid_ratio >= old_valid_ratio * min_valid_ratio_keep
        if not improvement_ok or not coverage_ok:
            break
        current_polygon = best_cut['processed_polygon']
        current_measurements = best_cut['measurements']
        current_quality = best_cut['quality']
        accepted_cuts.append(best_cut)
        removed_pieces_all.extend(best_cut['removed_pieces'])
    widths_px = [measurement['width_px'] for measurement in current_measurements if measurement['width_px'] is not None]
    return {'processed_polygon': current_polygon, 'removed_pieces': removed_pieces_all, 'accepted_cuts': accepted_cuts, 'cut_lines': [cut['visual_cut_line'] for cut in accepted_cuts], 'measurements': current_measurements, 'widths_px': widths_px, 'quality': current_quality, 'num_cuts': len(accepted_cuts)}
