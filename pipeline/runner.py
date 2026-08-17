import time
from pathlib import Path

from pipeline.graph.load import load_graph_edges
from pipeline.graph.update import update_geojson_widths_gpd
from pipeline.inference.model import load_inference_model
from pipeline.inference.predict import run_model_on_images
from pipeline.progress import build_progress
from pipeline.stages import STAGES
from pipeline.tiles.cache import prepare_tile_cache_images
from pipeline.tiles.selection import (
    build_edge_tile_index,
    filter_edges_with_complete_tiles,
)
from pipeline.tiles.sources import create_tile_source
from pipeline.width.calculate import calculate_widths_by_edge
from pipeline.width.sqlite_progress import build_width_run_key


class PipelineRunner:
    def __init__(
        self,
        log_callback=None,
        stage_callback=None,
        progress_callback=None,
        stop_requested=None,
        pause_wait=None,
    ):
        self.log_callback = log_callback
        self.stage_callback = stage_callback
        self.progress_callback = progress_callback
        self.stop_requested = stop_requested
        self.pause_wait = pause_wait

    def log(self, text):
        if self.log_callback is not None:
            self.log_callback(text)

    def set_stage(self, stage):
        if self.stage_callback is not None:
            self.stage_callback(stage)

    def report_progress(
        self,
        stage,
        current=0,
        total=1,
        message="",
        fraction_override=None,
    ):
        if self.progress_callback is None:
            return
        self.progress_callback(
            build_progress(
                stage,
                current=current,
                total=total,
                message=message,
                fraction_override=fraction_override,
            ).as_dict()
        )

    def report_fraction(
        self,
        stage,
        fraction,
        message="",
        current=0,
        total=0,
    ):
        fraction = min(max(float(fraction), 0.0), 1.0)
        self.report_progress(
            stage,
            current=current,
            total=total,
            message=message,
            fraction_override=fraction,
        )

    def should_stop(self):
        return self.stop_requested is not None and self.stop_requested()

    def checkpoint(self):
        if self.should_stop():
            return False

        if self.pause_wait is not None and not self.pause_wait():
            return False

        return not self.should_stop()

    def run(
        self,
        graph_path,
        weights_path,
        source_type,
        input_tiles_dir,
        work_dir,
        output_geojson_path,
        z,
        context_px,
        tile_size,
        tile_ext,
        skip_existing,
        device_preference="cuda",
    ):
        total_started_at = time.perf_counter()
        stage_times = {}

        if not self.checkpoint():
            return {"cancelled": True}

        stage = STAGES[0]
        self.set_stage(stage)
        self.report_progress(stage, 0, 1, "Загрузка GeoJSON")
        self.log(f"Граф: {graph_path}")

        started_at = time.perf_counter()
        all_edges = load_graph_edges(graph_path)
        stage_times["load_graph"] = time.perf_counter() - started_at
        self.report_progress(
            stage,
            1,
            1,
            f"Загружено рёбер: {len(all_edges)}",
        )
        self.log(f"Считано рёбер: {len(all_edges)}")

        if not self.checkpoint():
            return {"cancelled": True}

        stage = STAGES[1]
        self.set_stage(stage)
        started_at = time.perf_counter()

        all_edge_tile_index, all_unique_tiles = build_edge_tile_index(
            edges=all_edges,
            z=z,
            context_px=context_px,
            tile_size=tile_size,
            progress_callback=lambda current, total: self.report_progress(
                stage,
                current,
                total,
                f"Рёбра: {current}/{total}",
            ),
        )

        stage_times["tile_indices"] = time.perf_counter() - started_at
        self.log(f"Уникальных требуемых тайлов: {len(all_unique_tiles)}")

        if not self.checkpoint():
            return {"cancelled": True}

        stage = STAGES[2]
        self.set_stage(stage)
        started_at = time.perf_counter()

        tile_source = create_tile_source(
            source_type=source_type,
            input_tiles_dir=input_tiles_dir,
            work_dir=work_dir,
        )

        missing_tiles = tile_source.find_missing(
            all_unique_tiles,
            progress_callback=lambda current, total: self.report_progress(
                stage,
                current,
                total,
                f"Проверено тайлов: {current}/{total}",
            ),
        )

        tile_filter = filter_edges_with_complete_tiles(
            edges=all_edges,
            edge_tile_index=all_edge_tile_index,
            missing_tiles=missing_tiles,
        )

        edges = tile_filter["eligible_edges"]
        edge_tile_index = tile_filter["edge_tile_index"]
        unique_tiles = tile_filter["unique_tiles"]
        skipped_edges = tile_filter["skipped_edges"]

        stage_times["find_tiles"] = time.perf_counter() - started_at

        self.log(f"Отсутствующих тайлов: {len(missing_tiles)}")
        self.log(
            "Рёбер исключено из разметки из-за неполного набора тайлов: "
            f"{len(skipped_edges)}"
        )
        self.log(f"Рёбер оставлено для разметки: {len(edges)}")
        self.log(f"Тайлов оставлено для разметки: {len(unique_tiles)}")

        if not self.checkpoint():
            return {"cancelled": True}

        stage = STAGES[3]
        self.set_stage(stage)
        started_at = time.perf_counter()

        cache_images = prepare_tile_cache_images(
            unique_tiles=unique_tiles,
            tile_source=tile_source,
            work_dir=work_dir,
            tile_ext=tile_ext,
            skip_existing=skip_existing,
            pause_wait=self.pause_wait,
            stop_requested=self.stop_requested,
            progress_callback=lambda current, total: self.report_progress(
                stage,
                current,
                total,
                f"Подготовлено тайлов: {current}/{total}",
            ),
        )

        if cache_images["cancelled"] or not self.checkpoint():
            return {"cancelled": True}

        # Между проверкой и копированием файл может быть удалён другим
        # процессом. В таком случае повторно исключаем только связанные
        # с ним рёбра, а не завершаем весь запуск ошибкой.
        if cache_images["missing"]:
            cache_missing_tiles = [
                item["tile"]
                for item in cache_images["missing"]
            ]
            second_filter = filter_edges_with_complete_tiles(
                edges=edges,
                edge_tile_index=edge_tile_index,
                missing_tiles=cache_missing_tiles,
            )
            edges = second_filter["eligible_edges"]
            edge_tile_index = second_filter["edge_tile_index"]
            unique_tiles = second_filter["unique_tiles"]
            skipped_edges.extend(second_filter["skipped_edges"])

            allowed_tile_keys = set(unique_tiles.keys())
            cache_images["image_items"] = [
                item
                for item in cache_images["image_items"]
                if (item["z"], item["x"], item["y"])
                in allowed_tile_keys
            ]
            cache_images["image_paths"] = [
                item["path"]
                for item in cache_images["image_items"]
            ]

            self.log(
                "Дополнительно исключено рёбер после подготовки кэша: "
                f"{len(second_filter['skipped_edges'])}"
            )

        stage_times["prepare_tiles"] = time.perf_counter() - started_at

        if cache_images["direct"]:
            self.log(
                "Локальных тайлов используется напрямую: "
                f"{len(cache_images['direct'])}"
            )
        else:
            self.log(f"Добавлено в кеш: {len(cache_images['copied'])}")
            self.log(f"Уже было в кеше: {len(cache_images['skipped'])}")
            self.log(
                f"Папка кэша: "
                f"{cache_images['images_dir'].resolve().as_posix()}"
            )

        self.log(f"Не удалось получить: {len(cache_images['missing'])}")

        labels_dir = Path(work_dir) / "tile_cache" / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        if edges:
            stage = STAGES[4]
            self.set_stage(stage)
            self.report_fraction(stage, 0.0, "Инициализация модели")
            started_at = time.perf_counter()

            model, device = load_inference_model(
                weights_path=weights_path,
                num_classes=4,
                device_preference="auto",
                log_callback=self.log,
            )
            self.report_fraction(stage, 0.05, f"Модель загружена на {device}")

            if not self.checkpoint():
                return {"cancelled": True}

            def prediction_progress(current, total):
                fraction = current / max(1, total)
                self.report_fraction(
                    stage,
                    0.05 + 0.95 * fraction,
                    f"Сегментация тайлов: {current}/{total}",
                    current=current,
                    total=total,
                )

            mask_result = run_model_on_images(
                model=model,
                device=device,
                images_dir=cache_images["images_dir"],
                labels_dir=labels_dir,
                skip_existing=skip_existing,
                pause_wait=self.pause_wait,
                stop_requested=self.stop_requested,
                progress_callback=prediction_progress,
                image_items=cache_images["image_items"],
            )

            if mask_result["cancelled"] or not self.checkpoint():
                return {"cancelled": True}

            stage_times["predict_masks"] = time.perf_counter() - started_at

            processed_images = (
                len(mask_result["saved"])
                + len(mask_result["skipped"])
            )

            self.log(f"Обработано изображений: {processed_images}")

            if mask_result["failed"]:
                self.log(
                    "Ошибок обработки изображений: "
                    f"{len(mask_result['failed'])}"
                )

            device_type = device.type
            del model
            if device_type == "cuda":
                import torch

                torch.cuda.empty_cache()

            stage = STAGES[5]
            self.set_stage(stage)
            started_at = time.perf_counter()

            width_progress_db_path = (
                Path(work_dir)
                / "width_progress.sqlite3"
            )
            width_run_key = build_width_run_key(
                graph_path=graph_path,
                weights_path=weights_path,
                z=z,
                context_px=context_px,
                tile_size=tile_size,
            )

            width_result = calculate_widths_by_edge(
                edges=edges,
                edge_tile_index=edge_tile_index,
                cache_images_dir=cache_images["images_dir"],
                cache_labels_dir=mask_result["labels_dir"],
                z=z,
                tile_size=tile_size,
                log_callback=self.log,
                stop_requested=self.stop_requested,
                pause_wait=self.pause_wait,
                progress_callback=lambda current, total: self.report_progress(
                    stage,
                    current,
                    total,
                    f"Обработано рёбер: {current}/{total}",
                ),
                progress_db_path=width_progress_db_path,
                progress_run_key=width_run_key,
            )

            if width_result["cancelled"] or not self.checkpoint():
                return {"cancelled": True}

            stage_times["calculate_widths"] = (
                time.perf_counter() - started_at
            )
        else:
            self.log(
                "Нет рёбер с полным набором тайлов. "
                "Сегментация и вычисление ширины пропущены."
            )

            stage = STAGES[4]
            self.set_stage(stage)
            self.report_progress(stage, 1, 1, "Нет тайлов для сегментации")
            stage_times["predict_masks"] = 0.0
            mask_result = {
                "cancelled": False,
                "labels_dir": labels_dir,
                "saved": [],
                "skipped": [],
                "failed": [],
            }

            stage = STAGES[5]
            self.set_stage(stage)
            self.report_progress(stage, 1, 1, "Нет рёбер для расчёта")
            stage_times["calculate_widths"] = 0.0
            width_result = {
                "cancelled": False,
                "width_results": [],
                "edge_results": [],
            }

        calculated = sum(
            1
            for item in width_result["width_results"]
            if item.get("width_m") is not None
        )
        failed = len(width_result["width_results"]) - calculated

        self.log(f"Результатов ширины: {len(width_result['width_results'])}")
        self.log(f"Ширина вычислена: {calculated}")
        self.log(f"Не удалось вычислить: {failed}")

        if not self.checkpoint():
            return {"cancelled": True}

        stage = STAGES[6]
        self.set_stage(stage)
        self.report_progress(stage, 0, 1, "Запись GeoJSON")
        started_at = time.perf_counter()

        output_geojson_path = output_geojson_path or None

        updated_gdf = update_geojson_widths_gpd(
            input_geojson_path=graph_path,
            output_geojson_path=output_geojson_path,
            width_results=width_result["width_results"],
            width_field="Width",
        )

        stage_times["update_graph"] = time.perf_counter() - started_at
        saved_graph_path = Path(
            output_geojson_path or graph_path
        ).resolve().as_posix()

        self.log(f"Граф сохранён: {saved_graph_path}")
        self.report_progress(stage, 1, 1, "Результат сохранён")

        return {
            "cancelled": False,
            "edges": edges,
            "all_edges": all_edges,
            "skipped_edges": skipped_edges,
            "edge_tile_index": edge_tile_index,
            "unique_tiles": unique_tiles,
            "missing_tiles": missing_tiles,
            "cache_images": cache_images,
            "mask_result": mask_result,
            "edge_results": width_result["edge_results"],
            "width_results": width_result["width_results"],
            "updated_gdf": updated_gdf,
            "output_geojson_path": saved_graph_path,
            "stage_times": stage_times,
            "elapsed_seconds": time.perf_counter() - total_started_at,
        }

