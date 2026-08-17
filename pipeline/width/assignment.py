from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import nearest_points

from pipeline.width.skeleton import (
    get_representative_line,
    score_edge_against_polygon_axis_by_dot,
)


def edge_to_linestring(edge):
    points = edge['pixel_points']
    if len(points) < 2:
        return None
    return LineString(points)


def assign_edge_to_sidewalk_polygon(edge_line, sidewalk_polygons, edge_buffer=10, max_distance=160, axis_weight=800.0, skeleton_pad=5, skeleton_min_line_length=3.0, axis_simplify_tolerance=8.0, axis_min_segment_length=8.0, edge_step=10, max_match_distance=120, min_parallel_cos=0.75, min_coverage_ratio=0.25):
    if edge_line is None or edge_line.is_empty:
        return (None, {'status': 'empty_edge', 'score': None, 'base_score': None, 'distance': None, 'axis_info': None, 'translated_edge': None})
    best_poly = None
    best_score = -1e+18
    best_info = None
    edge_area = edge_line.buffer(edge_buffer)
    for poly_idx, poly in enumerate(sidewalk_polygons):
        if poly is None or poly.is_empty:
            continue
        distance = edge_line.distance(poly)
        if distance > max_distance:
            continue
        intersection = edge_area.intersection(poly)
        intersection_area = 0.0 if intersection.is_empty else intersection.area
        line_intersection = edge_line.intersection(poly)
        intersection_length = 0.0 if line_intersection.is_empty else line_intersection.length
        base_score = intersection_area + intersection_length * 10.0 - distance * 0.5
        axis_info = score_edge_against_polygon_axis_by_dot(edge_line=edge_line, polygon=poly, skeleton_pad=skeleton_pad, skeleton_min_line_length=skeleton_min_line_length, axis_simplify_tolerance=axis_simplify_tolerance, axis_min_segment_length=axis_min_segment_length, edge_step=edge_step, max_match_distance=max_match_distance, min_parallel_cos=min_parallel_cos, distance_sigma=max_match_distance)
        if axis_info['coverage_ratio'] < min_coverage_ratio:
            score = base_score - axis_weight * 0.5
        else:
            score = base_score + axis_weight * axis_info['axis_score']
        if score > best_score:
            best_score = score
            best_poly = poly
            best_info = {'status': 'assigned', 'polygon_index': poly_idx, 'score': score, 'base_score': base_score, 'distance': distance, 'intersection_area': intersection_area, 'intersection_length': intersection_length, 'axis_score': axis_info['axis_score'], 'mean_dot': axis_info['mean_dot'], 'coverage_ratio': axis_info['coverage_ratio'], 'mean_distance': axis_info['mean_distance'], 'axis_status': axis_info['axis_status'], 'shift_dx': axis_info['shift_dx'], 'shift_dy': axis_info['shift_dy'], 'shift_length': axis_info['shift_length'], 'shift_std': axis_info['shift_std'], 'translated_edge': axis_info['translated_edge'], 'axis_info': axis_info}
    if best_poly is None:
        return (None, {'status': 'no_polygon', 'score': None, 'base_score': None, 'distance': None, 'axis_info': None, 'translated_edge': None})
    return (best_poly, best_info)


def snap_point_to_polygon(point_xy, polygon):
    point = Point(point_xy)
    if polygon.contains(point):
        return (point_xy, False)
    nearest_on_polygon = nearest_points(point, polygon.boundary)[1]
    return ((nearest_on_polygon.x, nearest_on_polygon.y), True)


def snap_edge_points_to_polygon(edge_line, polygon):
    snapped_points = []
    snap_flags = []
    for x, y in edge_line.coords:
        snapped, was_snapped = snap_point_to_polygon((x, y), polygon)
        snapped_points.append(snapped)
        snap_flags.append(was_snapped)
    return (LineString(snapped_points), snap_flags)


def get_edge_part_inside_polygon(edge_line, polygon):
    if edge_line is None or edge_line.is_empty:
        return None
    if polygon is None or polygon.is_empty:
        return None
    intersection = edge_line.intersection(polygon)
    if intersection.is_empty:
        return None
    if intersection.geom_type == 'LineString':
        if intersection.length > 0:
            return intersection
        return None
    if intersection.geom_type == 'MultiLineString':
        lines = [geom for geom in intersection.geoms if geom.length > 0]
        if len(lines) == 0:
            return None
        if len(lines) == 1:
            return lines[0]
        return MultiLineString(lines)
    if intersection.geom_type == 'GeometryCollection':
        lines = []
        for geom in intersection.geoms:
            if geom.geom_type == 'LineString' and geom.length > 0:
                lines.append(geom)
        if len(lines) == 0:
            return None
        if len(lines) == 1:
            return lines[0]
        return MultiLineString(lines)
    return None


def line_inside_polygon_ratio(line, polygon, tolerance=2.0):
    line = get_representative_line(line)
    if line is None or line.is_empty or line.length == 0:
        return 0.0
    polygon_check = polygon.buffer(tolerance)
    intersection = line.intersection(polygon_check)
    if intersection.is_empty:
        return 0.0
    if intersection.geom_type == 'LineString':
        inside_length = intersection.length
    elif intersection.geom_type == 'MultiLineString':
        inside_length = sum((part.length for part in intersection.geoms))
    elif intersection.geom_type == 'GeometryCollection':
        inside_length = sum((part.length for part in intersection.geoms if part.geom_type == 'LineString'))
    else:
        inside_length = 0.0
    return inside_length / line.length


def find_crosswalk_direct_target(edge_line, crosswalk_polygons, min_crosswalk_ratio=0.3, tolerance=0.0):
    edge_line = get_representative_line(edge_line)
    if edge_line is None or edge_line.is_empty or edge_line.length == 0:
        return {'use_crosswalk': False, 'crosswalk_ratio': 0.0, 'crosswalk_polygon': None, 'crosswalk_line': None}
    best_polygon = None
    best_ratio = 0.0
    for polygon in crosswalk_polygons:
        if polygon is None or polygon.is_empty:
            continue
        ratio = line_inside_polygon_ratio(line=edge_line, polygon=polygon, tolerance=tolerance)
        if ratio > best_ratio:
            best_ratio = ratio
            best_polygon = polygon
    if best_polygon is None or best_ratio <= min_crosswalk_ratio:
        return {'use_crosswalk': False, 'crosswalk_ratio': best_ratio, 'crosswalk_polygon': None, 'crosswalk_line': None}
    crosswalk_line = get_edge_part_inside_polygon(edge_line=edge_line, polygon=best_polygon)
    crosswalk_line = get_representative_line(crosswalk_line)
    if crosswalk_line is None or crosswalk_line.is_empty or crosswalk_line.length == 0:
        return {'use_crosswalk': False, 'crosswalk_ratio': best_ratio, 'crosswalk_polygon': best_polygon, 'crosswalk_line': None}
    return {'use_crosswalk': True, 'crosswalk_ratio': best_ratio, 'crosswalk_polygon': best_polygon, 'crosswalk_line': crosswalk_line}


def process_edge_for_sidewalk_width(edge, sidewalk_polygons, edge_buffer=8, max_distance=40, step=10, max_width=100, fallback_to_nearest=True, min_translated_inside_ratio=0.5, translated_inside_tolerance=2.0):
    edge_line = edge_to_linestring(edge)
    polygon, assign_info = assign_edge_to_sidewalk_polygon(edge_line=edge_line, sidewalk_polygons=sidewalk_polygons, edge_buffer=edge_buffer, max_distance=max_distance)
    if polygon is None:
        return {'edge': edge, 'edge_line': edge_line, 'translated_edge_line': None, 'polygon': None, 'assign_info': assign_info, 'measurements': [], 'snapped_line': None, 'work_line': None, 'snap_mode': 'no_polygon'}
    translated_edge_line = assign_info.get('translated_edge')
    if translated_edge_line is None and assign_info.get('axis_info') is not None:
        translated_edge_line = assign_info['axis_info'].get('translated_edge')
    translated_inside_ratio = None
    if translated_edge_line is not None and (not translated_edge_line.is_empty):
        translated_inside_ratio = line_inside_polygon_ratio(line=translated_edge_line, polygon=polygon, tolerance=translated_inside_tolerance)
        if translated_inside_ratio < min_translated_inside_ratio:
            return {'edge': edge, 'edge_line': edge_line, 'translated_edge_line': translated_edge_line, 'polygon': polygon, 'assign_info': assign_info, 'measurements': [], 'snapped_line': None, 'work_line': None, 'snap_mode': 'axis_translation_rejected', 'skip_reason': 'translated_edge_mostly_outside_polygon', 'translated_inside_ratio': translated_inside_ratio, 'min_translated_inside_ratio': min_translated_inside_ratio}
        work_line = translated_edge_line
        snap_mode = 'axis_translation'
    else:
        inside_line = get_edge_part_inside_polygon(edge_line=edge_line, polygon=polygon)
        if inside_line is not None:
            work_line = inside_line
            snap_mode = 'intersection'
        elif fallback_to_nearest:
            work_line, _ = snap_edge_points_to_polygon(edge_line=edge_line, polygon=polygon)
            snap_mode = 'nearest_boundary'
        else:
            return {'edge': edge, 'edge_line': edge_line, 'translated_edge_line': None, 'polygon': polygon, 'assign_info': assign_info, 'measurements': [], 'snapped_line': None, 'work_line': None, 'snap_mode': 'no_intersection'}
    return {'edge': edge, 'edge_line': edge_line, 'translated_edge_line': translated_edge_line, 'polygon': polygon, 'assign_info': assign_info, 'measurements': [], 'snapped_line': work_line, 'work_line': work_line, 'snap_mode': snap_mode, 'translated_inside_ratio': translated_inside_ratio, 'min_translated_inside_ratio': min_translated_inside_ratio}


def strongly_simplify_assigned_polygon(edge, sidewalk_polygons, simplify_tolerance, edge_buffer, max_distance, step, max_width, fallback_to_nearest):
    edge_result = process_edge_for_sidewalk_width(edge=edge, sidewalk_polygons=sidewalk_polygons, edge_buffer=edge_buffer, max_distance=max_distance, step=step, max_width=max_width, fallback_to_nearest=fallback_to_nearest)
    if edge_result.get('skip_reason') is not None:
        return {'status': edge_result['skip_reason'], 'edge_result': edge_result, 'polygon': edge_result.get('polygon'), 'simplified_polygon': None}
    polygon = edge_result['polygon']
    if polygon is None:
        return {'status': 'no_polygon', 'edge_result': edge_result, 'polygon': None, 'simplified_polygon': None}
    simplified_polygon = polygon.simplify(tolerance=simplify_tolerance, preserve_topology=True)
    if not simplified_polygon.is_valid:
        simplified_polygon = simplified_polygon.buffer(0)
    return {'status': 'ok', 'edge_result': edge_result, 'polygon': polygon, 'simplified_polygon': simplified_polygon}


def get_width_work_line_from_edge_result(edge_result):
    translated_edge_line = edge_result.get('translated_edge_line')
    if translated_edge_line is not None and (not translated_edge_line.is_empty):
        return translated_edge_line
    work_line = edge_result.get('work_line')
    if work_line is not None and (not work_line.is_empty):
        return work_line
    snapped_line = edge_result.get('snapped_line')
    if snapped_line is not None and (not snapped_line.is_empty):
        return snapped_line
    return edge_result.get('edge_line')
