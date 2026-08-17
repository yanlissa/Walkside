import numpy as np
from shapely.geometry import LineString

from pipeline.width.skeleton import get_direction_at_distance, get_representative_line


def estimate_width_from_narrow_measurements(measurements, pixel_size_m=0.08, lower_quantile=0.05, upper_quantile=0.35, min_count=3, round_digits=2):
    rows = []
    for idx, measurement in enumerate(measurements):
        width_px = measurement.get('width_px')
        if width_px is None:
            continue
        width_px = float(width_px)
        if not np.isfinite(width_px):
            continue
        rows.append({'index': idx, 'width_px': width_px})
    if not rows:
        return {'width_px': None, 'width_m': None, 'selected_indices': [], 'selected_widths_px': [], 'all_widths_px': [], 'method': 'narrow_quantile'}
    widths = np.array([row['width_px'] for row in rows], dtype=float)
    indices = np.array([row['index'] for row in rows], dtype=int)
    if len(widths) < min_count:
        selected_widths = widths
        selected_indices = indices
    else:
        q_low = np.quantile(widths, lower_quantile)
        q_high = np.quantile(widths, upper_quantile)
        mask = (widths >= q_low) & (widths <= q_high)
        selected_widths = widths[mask]
        selected_indices = indices[mask]
        if len(selected_widths) < min_count:
            mask = widths <= q_high
            selected_widths = widths[mask]
            selected_indices = indices[mask]
        if len(selected_widths) < min_count:
            order = np.argsort(widths)
            take_count = min(min_count, len(widths))
            selected_widths = widths[order[:take_count]]
            selected_indices = indices[order[:take_count]]
    width_px = float(np.median(selected_widths))
    width_m = round(width_px * pixel_size_m, round_digits)
    return {'width_px': width_px, 'width_m': width_m, 'selected_indices': selected_indices.tolist(), 'selected_widths_px': selected_widths.tolist(), 'all_widths_px': widths.tolist(), 'method': 'narrow_quantile', 'lower_quantile': lower_quantile, 'upper_quantile': upper_quantile}


def iter_lines_from_intersection(geom):
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == 'LineString':
        yield geom
    elif geom.geom_type == 'MultiLineString':
        for part in geom.geoms:
            if not part.is_empty:
                yield part
    elif geom.geom_type == 'GeometryCollection':
        for part in geom.geoms:
            yield from iter_lines_from_intersection(part)


def width_at_point_by_center_segment(polygon, center_point, direction, max_width):
    vx, vy = direction
    nx = -vy
    ny = vx
    cx = center_point.x
    cy = center_point.y
    p1 = (cx - nx * max_width, cy - ny * max_width)
    p2 = (cx + nx * max_width, cy + ny * max_width)
    full_normal_line = LineString([p1, p2])
    intersection = full_normal_line.intersection(polygon)
    if intersection.is_empty:
        return (None, full_normal_line, intersection)
    segments = list(iter_lines_from_intersection(intersection))
    if not segments:
        return (None, full_normal_line, intersection)
    best_segment = None
    best_distance = float('inf')
    for segment in segments:
        coords = list(segment.coords)
        if len(coords) < 2:
            continue
        t_values = []
        for x, y in coords:
            dx = x - cx
            dy = y - cy
            t = dx * nx + dy * ny
            t_values.append(t)
        t_min = min(t_values)
        t_max = max(t_values)
        if t_min <= 0 <= t_max:
            best_segment = segment
            break
        distance_to_center = min(abs(t_min), abs(t_max))
        if distance_to_center < best_distance:
            best_distance = distance_to_center
            best_segment = segment
    if best_segment is None:
        return (None, full_normal_line, intersection)
    return (best_segment.length, full_normal_line, best_segment)


def measure_width_along_edge_with_processed_polygon(edge_line, processed_polygon, step, max_width):
    edge_line = get_representative_line(edge_line)
    measurements = []
    if edge_line is None or edge_line.length == 0:
        return measurements
    distances = np.arange(0, edge_line.length + 1e-06, step)
    for distance in distances:
        center = edge_line.interpolate(float(distance))
        direction = get_direction_at_distance(line=edge_line, distance=float(distance))
        if direction is None:
            continue
        width_px, full_normal_line, width_segment = width_at_point_by_center_segment(polygon=processed_polygon, center_point=center, direction=direction, max_width=max_width)
        measurements.append({'distance_along_edge': float(distance), 'center': (center.x, center.y), 'width_px': width_px, 'full_normal_line': full_normal_line, 'intersection': width_segment})
    return measurements


def width_quality(measurements):
    widths = np.asarray([measurement['width_px'] for measurement in measurements if measurement['width_px'] is not None], dtype=float)
    if len(measurements) == 0 or len(widths) == 0:
        return {'score': -1e+18, 'mean_width': None, 'mae': None, 'rel_mae': float('inf'), 'valid_ratio': 0.0, 'valid_count': 0, 'total_count': len(measurements)}
    mean_width = float(np.mean(widths))
    mae = float(np.mean(np.abs(widths - mean_width)))
    rel_mae = mae / max(mean_width, 1e-06)
    valid_ratio = len(widths) / len(measurements)
    score = -rel_mae + 0.15 * valid_ratio
    return {'score': float(score), 'mean_width': mean_width, 'mae': mae, 'rel_mae': float(rel_mae), 'valid_ratio': float(valid_ratio), 'valid_count': int(len(widths)), 'total_count': int(len(measurements))}


def calculate_crosswalk_width_for_edge(edge, edge_line, crosswalk_polygon, crosswalk_line, crosswalk_ratio, step=10, max_width=120, pixel_size_m=0.08, min_width_m=3.5):
    measurements = measure_width_along_edge_with_processed_polygon(edge_line=crosswalk_line, processed_polygon=crosswalk_polygon, step=step, max_width=max_width)
    widths_px = [measurement['width_px'] for measurement in measurements if measurement['width_px'] is not None]
    quality = width_quality(measurements)
    multi_cut_result = {'processed_polygon': crosswalk_polygon, 'removed_pieces': [], 'accepted_cuts': [], 'cut_lines': [], 'measurements': measurements, 'widths_px': widths_px, 'quality': quality, 'num_cuts': 0}
    edge_result = {'edge': edge, 'edge_line': edge_line, 'translated_edge_line': None, 'polygon': crosswalk_polygon, 'assign_info': {'status': 'crosswalk_direct', 'axis_info': None}, 'measurements': measurements, 'snapped_line': crosswalk_line, 'work_line': crosswalk_line, 'snap_mode': 'crosswalk_direct'}
    simplified_result = {'status': 'ok', 'edge_result': edge_result, 'polygon': crosswalk_polygon, 'simplified_polygon': crosswalk_polygon}
    if not widths_px:
        return {'edge': edge, 'status': 'no_measurements', 'width_m': None, 'width_px': None, 'width_mode': 'crosswalk_direct', 'crosswalk_ratio': crosswalk_ratio, 'measurements': measurements, 'edge_line': crosswalk_line, 'simplified_result': simplified_result, 'multi_cut_result': multi_cut_result}
    narrow_width_stats = estimate_width_from_narrow_measurements(measurements=measurements, pixel_size_m=pixel_size_m, lower_quantile=0.15, upper_quantile=0.65, min_count=3)
    width_px = narrow_width_stats['width_px']
    width_m = narrow_width_stats['width_m']
    if width_px is None:
        width_px = float(np.median(widths_px))
        width_m = round(width_px * pixel_size_m, 2)
    algorithm_width_px = width_px
    algorithm_width_m = width_m
    if width_m < min_width_m:
        width_m = float(min_width_m)
        width_px = width_m / pixel_size_m
    return {'edge': edge, 'status': 'ok', 'width_m': width_m, 'width_px': width_px, 'algorithm_width_m': algorithm_width_m, 'algorithm_width_px': algorithm_width_px, 'width_mode': 'crosswalk_direct', 'crosswalk_ratio': crosswalk_ratio, 'measurements': measurements, 'edge_line': crosswalk_line, 'simplified_result': simplified_result, 'multi_cut_result': multi_cut_result, 'processed_polygon': crosswalk_polygon, 'removed_pieces': [], 'cut_lines': [], 'num_cuts': 0, 'quality': quality, 'narrow_width_stats': narrow_width_stats}
