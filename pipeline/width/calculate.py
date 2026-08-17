from pipeline.width.assignment import (
    edge_to_linestring,
    find_crosswalk_direct_target,
    get_width_work_line_from_edge_result,
    strongly_simplify_assigned_polygon,
)
from pipeline.width.cuts import iterative_local_cuts
from pipeline.width.measurement import (
    calculate_crosswalk_width_for_edge,
    estimate_width_from_narrow_measurements,
)
from pipeline.width.mosaic import build_width_context_for_edge
from pipeline.width.polygons import build_better_polygons_from_mask_image
from pipeline.width.sqlite_progress import SQLiteWidthProgress

import numpy as np


STATUS_TEXT = {
    "no_polygon": "подходящий полигон не найден",
    "translated_edge_mostly_outside_polygon": "перенесённое ребро большей частью находится вне полигона",
    "no_measurements": "не удалось получить измерения ширины",
    "no_tiles": "для ребра не найдены тайлы",
    "no_results": "результат расчёта отсутствует",
}


def calculate_width_for_edge(edge, sidewalk_polygons, crosswalk_polygons=None, crosswalk_min_ratio=0.3, crosswalk_tolerance=0.0, edge_buffer=10, max_distance=50, step=10, max_width=120, fallback_to_nearest=True, strong_simplify_tolerance=20, angle_threshold=0.0, min_parallel_cos=0.75, split_extension=5, edge_buffer_for_keep=5, min_area_after_split=20, min_removed_area=50, max_removed_area_ratio=0.75, max_cuts=5, min_rel_mae_improvement=0.03, min_valid_ratio=0.5, min_valid_ratio_keep=0.8, local_direction_delta=25.0, max_cut_distance_to_edge=120.0, min_crosswalk_width_m=3.5, pixel_size_m=0.08):
    edge_line = edge_to_linestring(edge)
    crosswalk_target = {'use_crosswalk': False, 'crosswalk_ratio': 0.0, 'crosswalk_polygon': None, 'crosswalk_line': None}

    if crosswalk_polygons:
        crosswalk_target = find_crosswalk_direct_target(
            edge_line=edge_line,
            crosswalk_polygons=crosswalk_polygons,
            min_crosswalk_ratio=crosswalk_min_ratio,
            tolerance=crosswalk_tolerance,
        )

    if crosswalk_target['use_crosswalk']:
        return calculate_crosswalk_width_for_edge(
            edge=edge,
            edge_line=edge_line,
            crosswalk_polygon=crosswalk_target['crosswalk_polygon'],
            crosswalk_line=crosswalk_target['crosswalk_line'],
            crosswalk_ratio=crosswalk_target['crosswalk_ratio'],
            step=step,
            max_width=max_width,
            pixel_size_m=pixel_size_m,
            min_width_m=min_crosswalk_width_m,
        )

    simplified_result = strongly_simplify_assigned_polygon(
        edge=edge,
        sidewalk_polygons=sidewalk_polygons,
        simplify_tolerance=strong_simplify_tolerance,
        edge_buffer=edge_buffer,
        max_distance=max_distance,
        step=step,
        max_width=max_width,
        fallback_to_nearest=fallback_to_nearest,
    )

    if simplified_result['status'] != 'ok':
        return {
            'edge': edge,
            'status': simplified_result['status'],
            'width_m': None,
            'width_px': None,
            'width_mode': 'combined',
            'crosswalk_ratio': crosswalk_target['crosswalk_ratio'],
            'measurements': [],
            'simplified_result': simplified_result,
            'multi_cut_result': None,
        }

    original_polygon = simplified_result['polygon']
    edge_result = simplified_result['edge_result']
    work_line = get_width_work_line_from_edge_result(edge_result)

    multi_cut_result = iterative_local_cuts(
        polygon=original_polygon,
        edge_line=work_line,
        step=step,
        max_width=max_width,
        max_cuts=max_cuts,
        simplify_tolerance=strong_simplify_tolerance,
        angle_threshold=angle_threshold,
        min_parallel_cos=min_parallel_cos,
        split_extension=split_extension,
        edge_buffer=edge_buffer_for_keep,
        min_area_after_split=min_area_after_split,
        min_removed_area=min_removed_area,
        max_removed_area_ratio=max_removed_area_ratio,
        local_direction_delta=local_direction_delta,
        max_cut_distance_to_edge=max_cut_distance_to_edge,
        min_valid_ratio=min_valid_ratio,
        min_rel_mae_improvement=min_rel_mae_improvement,
        min_valid_ratio_keep=min_valid_ratio_keep,
    )

    widths_px = multi_cut_result['widths_px']

    if not widths_px:
        return {
            'edge': edge,
            'status': 'no_measurements',
            'width_m': None,
            'width_px': None,
            'width_mode': 'combined',
            'crosswalk_ratio': crosswalk_target['crosswalk_ratio'],
            'width_polygon': original_polygon,
            'width_line': work_line,
            'measurements': multi_cut_result['measurements'],
            'edge_line': work_line,
            'simplified_result': simplified_result,
            'multi_cut_result': multi_cut_result,
        }

    narrow_width_stats = estimate_width_from_narrow_measurements(
        measurements=multi_cut_result['measurements'],
        pixel_size_m=pixel_size_m,
        lower_quantile=0.15,
        upper_quantile=0.65,
        min_count=3,
    )

    width_px = narrow_width_stats['width_px']
    width_m = narrow_width_stats['width_m']

    if width_px is None:
        width_px = float(np.median(widths_px))
        width_m = round(width_px * pixel_size_m, 2)

    return {
        'edge': edge,
        'status': 'ok',
        'width_m': width_m,
        'width_px': width_px,
        'width_mode': 'combined',
        'crosswalk_ratio': crosswalk_target['crosswalk_ratio'],
        'width_polygon': original_polygon,
        'width_line': work_line,
        'measurements': multi_cut_result['measurements'],
        'edge_line': work_line,
        'simplified_result': simplified_result,
        'multi_cut_result': multi_cut_result,
        'processed_polygon': multi_cut_result['processed_polygon'],
        'removed_pieces': multi_cut_result['removed_pieces'],
        'cut_lines': multi_cut_result['cut_lines'],
        'num_cuts': multi_cut_result['num_cuts'],
        'quality': multi_cut_result['quality'],
        'narrow_width_stats': narrow_width_stats,
    }


def calculate_widths_for_edges(
    selected_edges,
    mask_mosaic,
    target_colors=((61, 61, 245), (255, 4, 4)),
    min_area=100,
    base_simplify_tolerance=4.0,
    smooth_radius=2,
    edge_buffer=10,
    max_distance=50,
    step=10,
    max_width=120,
    fallback_to_nearest=True,
    strong_simplify_tolerance=20,
    angle_threshold=0.0,
    min_parallel_cos=0.75,
    split_extension=5,
    edge_buffer_for_keep=5,
    min_area_after_split=20,
    min_removed_area=50,
    max_removed_area_ratio=0.75,
    max_cuts=5,
    min_rel_mae_improvement=0.03,
    min_valid_ratio=0.5,
    min_valid_ratio_keep=0.8,
    local_direction_delta=25.0,
    max_cut_distance_to_edge=120.0,
    pixel_size_m=0.08,
):
    sidewalk_polygons, sidewalk_mask, mask_rgb = build_better_polygons_from_mask_image(
        mask_img=mask_mosaic,
        target_colors=target_colors,
        min_area=min_area,
        simplify_tolerance=base_simplify_tolerance,
        smooth_radius=smooth_radius,
    )

    crosswalk_polygons, crosswalk_mask, _ = build_better_polygons_from_mask_image(
        mask_img=mask_mosaic,
        target_colors=((255, 4, 4),),
        min_area=min_area,
        simplify_tolerance=base_simplify_tolerance,
        smooth_radius=smooth_radius,
    )

    width_results = []

    for edge in selected_edges:
        width_result = calculate_width_for_edge(
            edge=edge,
            sidewalk_polygons=sidewalk_polygons,
            crosswalk_polygons=crosswalk_polygons,
            crosswalk_min_ratio=0.5,
            crosswalk_tolerance=2.0,
            edge_buffer=edge_buffer,
            max_distance=max_distance,
            step=step,
            max_width=max_width,
            fallback_to_nearest=fallback_to_nearest,
            strong_simplify_tolerance=strong_simplify_tolerance,
            angle_threshold=angle_threshold,
            min_parallel_cos=min_parallel_cos,
            split_extension=split_extension,
            edge_buffer_for_keep=edge_buffer_for_keep,
            min_area_after_split=min_area_after_split,
            min_removed_area=min_removed_area,
            max_removed_area_ratio=max_removed_area_ratio,
            max_cuts=max_cuts,
            min_rel_mae_improvement=min_rel_mae_improvement,
            min_valid_ratio=min_valid_ratio,
            min_valid_ratio_keep=min_valid_ratio_keep,
            local_direction_delta=local_direction_delta,
            max_cut_distance_to_edge=max_cut_distance_to_edge,
            pixel_size_m=pixel_size_m,
        )

        width_results.append(width_result)

    return {
        "width_results": width_results,
        "sidewalk_polygons": sidewalk_polygons,
        "sidewalk_mask": sidewalk_mask,
        "crosswalk_polygons": crosswalk_polygons,
        "crosswalk_mask": crosswalk_mask,
        "mask_rgb": mask_rgb,
    }


def _compact_width_result(result, edge):
    return {
        "edge": edge,
        "status": result.get("status", "no_results"),
        "width_m": result.get("width_m"),
        "width_px": result.get("width_px"),
        "width_mode": result.get("width_mode"),
        "crosswalk_ratio": result.get("crosswalk_ratio"),
    }


def _compact_widths(width_results):
    return {
        "width_results": width_results,
        "sidewalk_polygons": None,
        "sidewalk_mask": None,
        "crosswalk_polygons": None,
        "crosswalk_mask": None,
        "mask_rgb": None,
    }


def calculate_widths_by_edge(
    edges,
    edge_tile_index,
    cache_images_dir,
    cache_labels_dir,
    z=20,
    tile_size=256,
    mask_alpha=0.45,
    show_background=False,
    include_outside_edges=True,
    target_colors=((61, 61, 245), (255, 4, 4)),
    min_area=100,
    base_simplify_tolerance=4.0,
    smooth_radius=2,
    edge_buffer=10,
    max_distance=50,
    step=10,
    max_width=120,
    fallback_to_nearest=True,
    strong_simplify_tolerance=20,
    angle_threshold=0.0,
    min_parallel_cos=0.75,
    split_extension=5,
    edge_buffer_for_keep=5,
    min_area_after_split=20,
    min_removed_area=50,
    max_removed_area_ratio=0.75,
    max_cuts=5,
    min_rel_mae_improvement=0.03,
    min_valid_ratio=0.5,
    min_valid_ratio_keep=0.8,
    local_direction_delta=25.0,
    max_cut_distance_to_edge=120.0,
    pixel_size_m=0.08,
    log_callback=None,
    stop_requested=None,
    pause_wait=None,
    progress_callback=None,
    progress_db_path=None,
    progress_run_key=None,
):
    all_width_results = []
    edge_results = []

    progress_store = None
    if progress_db_path is not None and progress_run_key is not None:
        progress_store = SQLiteWidthProgress(progress_db_path)

        if log_callback is not None:
            cached_count = progress_store.count(progress_run_key)
            if cached_count > 0:
                log_callback(
                    f"В checkpoint найдено результатов ширины: {cached_count}"
                )

    def finish(cancelled):
        if progress_store is not None:
            progress_store.close()

        return {
            "cancelled": cancelled,
            "width_results": all_width_results,
            "edge_results": edge_results,
        }

    total_edges = len(edges)

    if progress_callback is not None:
        progress_callback(0, total_edges)

    for edge_index, edge in enumerate(edges):
        if pause_wait is not None and not pause_wait():
            return finish(True)

        if stop_requested is not None and stop_requested():
            return finish(True)

        edge_tiles = edge_tile_index.get(edge_index, [])
        edge_id = edge.get("properties", {}).get("EdgeId")

        if not edge_tiles:
            edge_results.append({
                "edge_index": edge_index,
                "edge": edge,
                "status": "no_tiles",
                "widths": None,
            })

            if log_callback is not None:
                log_callback(
                    f"Ребро {edge_index + 1}/{len(edges)}: "
                    f"EdgeId={edge_id}, тайлов=0, "
                    f"ширина не вычислена, причина: {STATUS_TEXT['no_tiles']}"
                )

            if progress_callback is not None:
                progress_callback(edge_index + 1, total_edges)
            continue

        cached_result = None
        if progress_store is not None:
            cached_result = progress_store.get(
                progress_run_key,
                edge["feature_idx"],
                edge["part_idx"],
            )

        if cached_result is not None:
            compact_result = {
                "edge": edge,
                **cached_result,
            }
            compact_widths = _compact_widths([compact_result])

            all_width_results.append(compact_result)
            edge_results.append({
                "edge_index": edge_index,
                "edge": edge,
                "status": "ok",
                "widths": compact_widths,
                "checkpoint": True,
            })

            if log_callback is not None:
                if compact_result.get("width_m") is not None:
                    log_callback(
                        f"Ребро {edge_index + 1}/{len(edges)}: "
                        f"EdgeId={edge_id}, тайлов={len(edge_tiles)}, "
                        f"ширина={compact_result['width_m']:.2f} м, checkpoint"
                    )
                else:
                    status = compact_result.get("status", "no_results")
                    status_text = STATUS_TEXT.get(status, status)
                    log_callback(
                        f"Ребро {edge_index + 1}/{len(edges)}: "
                        f"EdgeId={edge_id}, тайлов={len(edge_tiles)}, "
                        f"ширина не вычислена, причина: {status_text}, checkpoint"
                    )

            if progress_callback is not None:
                progress_callback(edge_index + 1, total_edges)
            continue

        width_context = build_width_context_for_edge(
            edge=edge,
            edge_tiles=edge_tiles,
            images_dir=cache_images_dir,
            labels_dir=cache_labels_dir,
            z=z,
            tile_size=tile_size,
            mask_alpha=mask_alpha,
            show_background=show_background,
            include_outside_edges=include_outside_edges,
        )

        if width_context["status"] != "ok":
            edge_results.append({
                "edge_index": edge_index,
                "edge": edge,
                "status": width_context["status"],
                "missing_images": width_context["missing_images"],
                "missing_labels": width_context["missing_labels"],
                "widths": None,
            })

            status = width_context["status"]
            status_text = STATUS_TEXT.get(status, status)

            if log_callback is not None:
                log_callback(
                    f"Ребро {edge_index + 1}/{len(edges)}: "
                    f"EdgeId={edge_id}, тайлов={len(edge_tiles)}, "
                    f"ширина не вычислена, причина: {status_text}"
                )

            if progress_callback is not None:
                progress_callback(edge_index + 1, total_edges)
            continue

        if pause_wait is not None and not pause_wait():
            return finish(True)

        if stop_requested is not None and stop_requested():
            return finish(True)

        widths = calculate_widths_for_edges(
            selected_edges=width_context["selected_edges"],
            mask_mosaic=width_context["mask_mosaic"],
            target_colors=target_colors,
            min_area=min_area,
            base_simplify_tolerance=base_simplify_tolerance,
            smooth_radius=smooth_radius,
            edge_buffer=edge_buffer,
            max_distance=max_distance,
            step=step,
            max_width=max_width,
            fallback_to_nearest=fallback_to_nearest,
            strong_simplify_tolerance=strong_simplify_tolerance,
            angle_threshold=angle_threshold,
            min_parallel_cos=min_parallel_cos,
            split_extension=split_extension,
            edge_buffer_for_keep=edge_buffer_for_keep,
            min_area_after_split=min_area_after_split,
            min_removed_area=min_removed_area,
            max_removed_area_ratio=max_removed_area_ratio,
            max_cuts=max_cuts,
            min_rel_mae_improvement=min_rel_mae_improvement,
            min_valid_ratio=min_valid_ratio,
            min_valid_ratio_keep=min_valid_ratio_keep,
            local_direction_delta=local_direction_delta,
            max_cut_distance_to_edge=max_cut_distance_to_edge,
            pixel_size_m=pixel_size_m,
        )

        edge_width_results = widths["width_results"]
        compact_results = [
            _compact_width_result(result, edge)
            for result in edge_width_results
        ]
        compact_widths = _compact_widths(compact_results)

        all_width_results.extend(compact_results)

        edge_results.append({
            "edge_index": edge_index,
            "edge": edge,
            "status": "ok",
            "widths": compact_widths,
        })

        if progress_store is not None and compact_results:
            progress_store.put(
                progress_run_key,
                edge,
                compact_results[0],
            )

        successful_result = next(
            (
                item
                for item in compact_results
                if item.get("width_m") is not None
            ),
            None,
        )

        if log_callback is not None:
            if successful_result is not None:
                log_callback(
                    f"Ребро {edge_index + 1}/{len(edges)}: "
                    f"EdgeId={edge_id}, тайлов={len(edge_tiles)}, "
                    f"ширина={successful_result['width_m']:.2f} м"
                )
            else:
                status = (
                    compact_results[0].get("status")
                    if compact_results
                    else "no_results"
                )

                status_text = STATUS_TEXT.get(status, status)

                log_callback(
                    f"Ребро {edge_index + 1}/{len(edges)}: "
                    f"EdgeId={edge_id}, тайлов={len(edge_tiles)}, "
                    f"ширина не вычислена, причина: {status_text}"
                )

        del widths
        del width_context
        del edge_width_results

        if progress_callback is not None:
            progress_callback(edge_index + 1, total_edges)

    return finish(False)
