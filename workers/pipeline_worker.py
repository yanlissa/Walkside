import threading
import traceback

from PySide6.QtCore import QObject, Signal, Slot

from pipeline.runner import PipelineRunner


class PipelineWorker(QObject):
    log = Signal(str)
    stage_changed = Signal(object)
    progress_changed = Signal(object)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    done = Signal()

    def __init__(
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
    ):
        super().__init__()

        self.graph_path = graph_path
        self.weights_path = weights_path
        self.source_type = source_type
        self.input_tiles_dir = input_tiles_dir
        self.work_dir = work_dir
        self.output_geojson_path = output_geojson_path
        self.z = z
        self.context_px = context_px
        self.tile_size = tile_size
        self.tile_ext = tile_ext
        self.skip_existing = skip_existing

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

    def request_pause(self):
        self._pause_event.set()

    def request_resume(self):
        self._pause_event.clear()

    def request_stop(self):
        self._stop_event.set()
        self._pause_event.clear()

    def is_stop_requested(self):
        return self._stop_event.is_set()

    def wait_if_paused(self):
        while self._pause_event.is_set():
            if self._stop_event.wait(0.1):
                return False

        return not self._stop_event.is_set()

    @Slot()
    def run(self):
        try:
            runner = PipelineRunner(
                log_callback=self.log.emit,
                stage_callback=self.stage_changed.emit,
                progress_callback=self.progress_changed.emit,
                stop_requested=self.is_stop_requested,
                pause_wait=self.wait_if_paused,
            )

            result = runner.run(
                graph_path=self.graph_path,
                weights_path=self.weights_path,
                source_type=self.source_type,
                input_tiles_dir=self.input_tiles_dir,
                work_dir=self.work_dir,
                output_geojson_path=self.output_geojson_path,
                z=self.z,
                context_px=self.context_px,
                tile_size=self.tile_size,
                tile_ext=self.tile_ext,
                skip_existing=self.skip_existing,
            )

            if result["cancelled"] or self.is_stop_requested():
                self.cancelled.emit()
            else:
                self.finished.emit(result)

        except Exception:
            self.failed.emit(traceback.format_exc())

        finally:
            self.done.emit()
