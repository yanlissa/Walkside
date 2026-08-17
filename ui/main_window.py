import time
from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import (
    APP_TITLE,
    SKIP_EXISTING,
    TILE_CONTEXT_PX,
    TILE_EXT,
    TILE_SIZE,
    ZOOM,
)

from pipeline.progress import EtaEstimator
from pipeline.stages import STAGES
from ui.styles import STYLE
from ui.widgets.path_row import PathRow
from ui.windows_taskbar import WindowsTaskbarProgress
from workers.pipeline_worker import PipelineWorker


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.edges = None
        self.edge_tile_index = None
        self.unique_tiles = None
        self.cache_images = None
        self.mask_result = None
        self.edge_results = None
        self.width_results = None
        self.updated_gdf = None
        self.output_geojson_result_path = None
        self.worker = None
        self.worker_thread = None
        self.started_at = None
        self.is_paused = False
        self.pause_started_at = None
        self.total_paused_seconds = 0.0
        self.eta_estimator = EtaEstimator()
        self.eta_seconds = None
        self.last_progress_percent = 0.0
        self.taskbar_progress = None

        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.setInterval(250)
        self.elapsed_timer.timeout.connect(self.update_elapsed)

        self.setWindowTitle(APP_TITLE)
        self.resize(1280, 720)
        self.setMinimumSize(860, 560)

        self.build_ui()
        self.setStyleSheet(STYLE)
        self.set_idle_state()
        self.log("Ожидание начала запуска.")

    def on_tile_source_changed(self):
        source_type = self.tile_source.currentData()
        self.tiles_dir.setEnabled(source_type == "local")

    def build_ui(self):
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        top_title = QFrame()
        top_title.setObjectName("topTitle")

        body = QFrame()
        body.setObjectName("body")

        main_layout.addWidget(top_title, 1)
        main_layout.addWidget(body, 9)

        title_layout = QVBoxLayout(top_title)
        title_layout.setContentsMargins(16, 10, 16, 10)
        title_layout.setSpacing(2)

        title = QLabel("WalkSide")
        title.setObjectName("title")

        subtitle = QLabel("Измерение ширины тротуаров")
        subtitle.setObjectName("muted")

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(14, 10, 14, 14)
        body_layout.setSpacing(0)

        self.body_splitter = QSplitter(Qt.Horizontal)
        self.body_splitter.setObjectName("bodySplitter")
        self.body_splitter.setChildrenCollapsible(False)
        self.body_splitter.setHandleWidth(10)

        self.files_path = self.build_left_panel()
        self.progress = self.build_right_panel()

        self.body_splitter.addWidget(self.files_path)
        self.body_splitter.addWidget(self.progress)
        self.body_splitter.setCollapsible(0, False)
        self.body_splitter.setCollapsible(1, False)
        self.body_splitter.setStretchFactor(0, 0)
        self.body_splitter.setStretchFactor(1, 1)
        self.body_splitter.setSizes([360, 900])

        body_layout.addWidget(self.body_splitter)

    def build_left_panel(self):
        panel = QFrame()
        panel.setObjectName("files_path")
        panel.setMinimumWidth(310)
        panel.setMaximumWidth(460)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        scroll = QScrollArea()
        scroll.setObjectName("leftScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.viewport().setObjectName("leftScrollViewport")

        inner = QWidget()
        inner.setObjectName("leftScrollInner")

        form = QVBoxLayout(inner)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(10)

        self.graph_path = PathRow(
            "Входной граф",
            "edges_with_width.geojson",
            "file",
        )

        self.weights_path = PathRow(
            "Веса модели",
            "model.pth",
            "file",
        )

        source_label = QLabel("Источник тайлов")
        source_label.setObjectName("pathTitle")

        self.tile_source = QComboBox()
        self.tile_source.addItem("Локальная папка", "local")
        self.tile_source.addItem("Внешний сервис", "remote")

        # Меняем только цвета раскрывающегося списка.
        # Размер и внешний вид закрытого QComboBox остаются прежними.
        self.tile_source.view().setStyleSheet(
            """
            QAbstractItemView {
                background-color: #111827;
                color: #f3f4f6;
                border: 1px solid #334155;
                outline: 0px;
                selection-background-color: #5b8cff;
                selection-color: #ffffff;
            }

            QAbstractItemView::item:disabled {
                background-color: #111827;
                color: #94a3b8;
            }
            """
        )

        self.tile_source.currentIndexChanged.connect(
            self.on_tile_source_changed
        )
        
        self.tiles_dir = PathRow(
            "Папка с тайлами",
            "all_tiles_ortophotoplan",
            "dir",
        )

        self.on_tile_source_changed()

        self.work_dir = PathRow(
            "Рабочая папка",
            "work_tiles",
            "dir",
        )

        self.output_path = PathRow(
            "Выходной GeoJSON",
            "result.geojson",
            "save",
        )

        form.addWidget(self.graph_path)
        form.addWidget(self.weights_path)
        form.addWidget(source_label)
        form.addWidget(self.tile_source)
        form.addWidget(self.tiles_dir)
        form.addWidget(self.work_dir)
        form.addWidget(self.output_path)
        form.addStretch(1)

        scroll.setWidget(inner)

        self.start_button = QPushButton("Запустить")
        self.start_button.setObjectName("primary")
        self.start_button.setMinimumHeight(46)
        self.start_button.clicked.connect(self.start_pipeline)

        self.pause_button = QPushButton("Пауза")
        self.pause_button.setMinimumHeight(42)
        self.pause_button.clicked.connect(self.toggle_pause)

        self.stop_button = QPushButton("Стоп")
        self.stop_button.setObjectName("danger")
        self.stop_button.setMinimumHeight(42)
        self.stop_button.clicked.connect(self.stop_pipeline)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(8)
        button_row.addWidget(self.pause_button, 1)
        button_row.addWidget(self.stop_button, 1)

        layout.addWidget(scroll, 1)
        layout.addWidget(self.start_button)
        layout.addLayout(button_row)

        return panel

    def build_right_panel(self):
        panel = QFrame()
        panel.setObjectName("progress")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        taskbar = QFrame()
        taskbar.setObjectName("taskbar")
        taskbar.setMaximumHeight(280)

        taskbar_layout = QVBoxLayout(taskbar)
        taskbar_layout.setContentsMargins(0, 0, 0, 0)
        taskbar_layout.setSpacing(10)

        self.state_title = QLabel("Готов к запуску")
        self.state_title.setObjectName("stateTitle")

        self.stage_label = QLabel("Пайплайн не запущен")
        self.stage_label.setObjectName("muted")

        self.overall_progress_label = QLabel("Общий прогресс")
        self.overall_progress_label.setObjectName("statTitle")

        self.progress_bar = QProgressBar()
        # 0..1000 позволяет показывать десятые доли процента и передавать
        # тот же масштаб в Windows taskbar.
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0%")
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(32)

        self.stage_progress_label = QLabel("Текущий этап")
        self.stage_progress_label.setObjectName("statTitle")

        self.stage_progress_bar = QProgressBar()
        self.stage_progress_bar.setRange(0, 1000)
        self.stage_progress_bar.setValue(0)
        self.stage_progress_bar.setFormat("0%")
        self.stage_progress_bar.setTextVisible(True)
        self.stage_progress_bar.setMinimumHeight(28)

        self.stage_progress_detail = QLabel("—")
        self.stage_progress_detail.setObjectName("muted")
        self.stage_progress_detail.setWordWrap(True)

        time_row = QHBoxLayout()
        time_row.setContentsMargins(0, 0, 0, 0)
        time_row.setSpacing(35)

        elapsed_widget, self.elapsed_value = self.build_stat_item("Прошло:")
        remaining_widget, self.remaining_value = self.build_stat_item("Осталось:")

        time_row.addWidget(elapsed_widget)
        time_row.addWidget(remaining_widget)
        time_row.addStretch(1)

        taskbar_layout.addWidget(self.state_title)
        taskbar_layout.addWidget(self.stage_label)
        taskbar_layout.addWidget(self.overall_progress_label)
        taskbar_layout.addWidget(self.progress_bar)
        taskbar_layout.addWidget(self.stage_progress_label)
        taskbar_layout.addWidget(self.stage_progress_bar)
        taskbar_layout.addWidget(self.stage_progress_detail)
        taskbar_layout.addLayout(time_row)

        logs = QFrame()
        logs.setObjectName("logs")

        logs_layout = QVBoxLayout(logs)
        logs_layout.setContentsMargins(0, 0, 0, 0)
        logs_layout.setSpacing(8)

        log_header = QHBoxLayout()
        log_header.setContentsMargins(0, 0, 0, 0)
        log_header.setSpacing(8)

        log_title = QLabel("Логи")
        log_title.setObjectName("sectionTitle")

        self.clear_log_button = QPushButton("Очистить")
        self.clear_log_button.clicked.connect(self.clear_logs)

        log_header.addWidget(log_title, 1)
        log_header.addWidget(self.clear_log_button)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        logs_layout.addLayout(log_header)
        logs_layout.addWidget(self.log_view, 1)

        layout.addWidget(taskbar)
        layout.addWidget(logs, 1)

        return panel

    def build_stat_item(self, title_text, value_text="—"):
        wrapper = QWidget()

        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QLabel(title_text)
        title.setObjectName("statTitle")
        title.setFixedWidth(82)

        value = QLabel(value_text)
        value.setObjectName("statValue")
        value.setMinimumWidth(120)
        value.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        layout.addWidget(title)
        layout.addWidget(value)

        return wrapper, value

    def set_idle_state(self):
        self.state_title.setText("Готов к запуску")
        self.stage_label.setText("Пайплайн не запущен")

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0%")
        self.stage_progress_bar.setValue(0)
        self.stage_progress_bar.setFormat("0%")
        self.stage_progress_detail.setText("—")

        self.elapsed_value.setText("—")
        self.remaining_value.setText("—")

        self.start_button.setEnabled(True)
        self.pause_button.setText("Пауза")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)

        if self.taskbar_progress is not None:
            self.taskbar_progress.clear()

    def start_pipeline(self):
        if self.worker_thread is not None and self.worker_thread.isRunning():
            return

        self.edges = None
        self.edge_tile_index = None
        self.unique_tiles = None
        self.cache_images = None
        self.mask_result = None
        self.edge_results = None
        self.width_results = None
        self.updated_gdf = None
        self.output_geojson_result_path = None
        self.is_paused = False
        self.pause_started_at = None
        self.total_paused_seconds = 0.0
        self.eta_estimator.reset()
        self.eta_seconds = None
        self.last_progress_percent = 0.0
        self.started_at = time.perf_counter()
        self.elapsed_timer.start()

        self.state_title.setText("Выполняется")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0%")
        self.stage_progress_bar.setValue(0)
        self.stage_progress_bar.setFormat("0%")
        self.stage_progress_detail.setText("Подготовка к запуску")
        self.elapsed_value.setText("0 сек")
        self.remaining_value.setText("Оценка…")

        if self.taskbar_progress is not None:
            self.taskbar_progress.set_indeterminate()

        self.start_button.setEnabled(False)
        self.pause_button.setText("Пауза")
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)

        self.worker_thread = QThread(self)
        self.worker = PipelineWorker(
            graph_path=self.graph_path.text(),
            weights_path=self.weights_path.text(),
            source_type=self.tile_source.currentData(),
            input_tiles_dir=self.tiles_dir.text(),
            work_dir=self.work_dir.text(),
            output_geojson_path=self.output_path.text(),
            z=ZOOM,
            context_px=TILE_CONTEXT_PX,
            tile_size=TILE_SIZE,
            tile_ext=TILE_EXT,
            skip_existing=SKIP_EXISTING,
        )
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)

        self.worker.log.connect(self.log)
        self.worker.stage_changed.connect(self.on_stage_changed)
        self.worker.progress_changed.connect(self.on_progress_changed)
        self.worker.finished.connect(self.on_pipeline_finished)
        self.worker.failed.connect(self.on_pipeline_failed)
        self.worker.cancelled.connect(self.on_pipeline_cancelled)
        self.worker.done.connect(self.worker_thread.quit)
        self.worker.done.connect(self.worker.deleteLater)

        self.worker_thread.finished.connect(self.on_worker_thread_finished)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.log("Запуск.")
        self.worker_thread.start()

    def toggle_pause(self):
        if self.worker is None:
            return

        if self.is_paused:
            self.worker.request_resume()
            if self.pause_started_at is not None:
                self.total_paused_seconds += time.perf_counter() - self.pause_started_at
            self.pause_started_at = None
            self.is_paused = False
            self.pause_button.setText("Пауза")
            self.state_title.setText("Выполняется")
            if self.taskbar_progress is not None:
                self.taskbar_progress.set_progress(round(self.last_progress_percent * 10))
            self.log("Выполнение продолжено.")
        else:
            self.worker.request_pause()
            self.pause_started_at = time.perf_counter()
            self.is_paused = True
            self.pause_button.setText("Продолжить")
            self.state_title.setText("Пауза")
            self.remaining_value.setText("Пауза")
            if self.taskbar_progress is not None:
                self.taskbar_progress.set_paused(round(self.last_progress_percent * 10))
            self.log("Выполнение приостановлено.")

    def stop_pipeline(self):
        if self.worker is None:
            return

        self.worker.request_stop()
        if self.pause_started_at is not None:
            self.total_paused_seconds += time.perf_counter() - self.pause_started_at
        self.pause_started_at = None
        self.is_paused = False
        self.pause_button.setText("Пауза")
        self.pause_button.setEnabled(False)
        self.state_title.setText("Остановка")
        self.stop_button.setEnabled(False)
        self.log("Запрошена остановка.")

    def on_stage_changed(self, stage):
        self.stage_label.setText(f"{stage.title}: {stage.description}")
        self.stage_progress_bar.setValue(0)
        self.stage_progress_bar.setFormat("0%")
        self.stage_progress_detail.setText(stage.description)

    def on_progress_changed(self, info):
        percent = min(
            max(float(info.get("overall_percent", 0.0)), 0.0),
            100.0,
        )

        # Не разрешаем общему прогрессу двигаться назад.
        self.last_progress_percent = max(
            self.last_progress_percent,
            percent,
        )

        # QProgressBar и Windows taskbar используют диапазон 0–1000.
        value = int(round(self.last_progress_percent * 10.0))

        # Общий прогресс всего pipeline.
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(
            f"{self.last_progress_percent:.1f}%"
        )

        # Отдельно показываем прогресс текущего этапа. Это особенно заметно
        # на длительных этапах подготовки/скачивания тайлов, где общий
        # прогресс занимает лишь небольшую часть шкалы всего pipeline.
        stage_percent = min(
            max(float(info.get("stage_percent", 0.0)), 0.0),
            100.0,
        )
        stage_value = int(round(stage_percent * 10.0))
        self.stage_progress_bar.setValue(stage_value)
        self.stage_progress_bar.setFormat(f"{stage_percent:.1f}%")

        current = max(0, int(info.get("current", 0) or 0))
        total = max(0, int(info.get("total", 0) or 0))
        message = str(info.get("message", "") or "").strip()

        if total > 0:
            count_text = f"{current:,} / {total:,}".replace(",", " ")
            detail_text = message or count_text
        else:
            detail_text = message or "Выполняется…"

        self.stage_progress_detail.setText(detail_text)

        # Прогресс на значке приложения в панели задач Windows.
        if self.taskbar_progress is not None:
            if self.is_paused:
                self.taskbar_progress.set_paused(value, 1000)
            else:
                self.taskbar_progress.set_progress(value, 1000)

        # Расчёт примерного оставшегося времени.
        elapsed = self.active_elapsed_seconds()

        if not self.is_paused and self.last_progress_percent > 0.0:
            self.eta_estimator.add_sample(
                elapsed,
                self.last_progress_percent,
            )

            self.eta_seconds = self.eta_estimator.estimate(
                elapsed,
                self.last_progress_percent,
            )

            if self.eta_seconds is None:
                self.remaining_value.setText("Оценка…")
            else:
                self.remaining_value.setText(
                    self.format_seconds(self.eta_seconds)
                )

    def on_pipeline_finished(self, result):
        self.elapsed_timer.stop()

        self.edges = result["edges"]
        self.edge_tile_index = result["edge_tile_index"]
        self.unique_tiles = result["unique_tiles"]
        self.cache_images = result["cache_images"]
        self.mask_result = result["mask_result"]
        self.edge_results = result["edge_results"]
        self.width_results = result["width_results"]
        self.updated_gdf = result["updated_gdf"]
        self.output_geojson_result_path = result["output_geojson_path"]

        self.progress_bar.setValue(1000)
        self.progress_bar.setFormat("100%")
        self.stage_progress_bar.setValue(1000)
        self.stage_progress_bar.setFormat("100%")
        self.stage_progress_detail.setText("Готово")
        self.state_title.setText("Завершено")
        self.stage_label.setText(f"{STAGES[6].title}: результат сохранён")

        self.elapsed_value.setText(
            self.format_seconds(result["elapsed_seconds"])
        )

        self.remaining_value.setText("0 сек")
        self.is_paused = False
        self.pause_started_at = None
        if self.taskbar_progress is not None:
            self.taskbar_progress.set_progress(1000)
            QTimer.singleShot(1500, self.taskbar_progress.clear)
        self.pause_button.setText("Пауза")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)

    def on_pipeline_failed(self, error_text):
        self.elapsed_timer.stop()
        self.state_title.setText("Ошибка")
        self.pause_started_at = None
        self.stage_label.setText("Выполнение завершилось с ошибкой")
        self.stage_progress_detail.setText("Ошибка на текущем этапе")
        self.is_paused = False
        self.pause_button.setText("Пауза")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.remaining_value.setText("—")
        if self.taskbar_progress is not None:
            self.taskbar_progress.set_error(round(self.last_progress_percent * 10))
        self.log(error_text)

    def on_pipeline_cancelled(self):
        self.elapsed_timer.stop()
        self.state_title.setText("Остановлено")
        self.pause_started_at = None
        self.stage_label.setText("Запуск прерван пользователем")
        self.stage_progress_detail.setText("Остановлено пользователем")
        self.is_paused = False
        self.pause_button.setText("Пауза")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.remaining_value.setText("—")
        if self.taskbar_progress is not None:
            self.taskbar_progress.clear()
        self.log("Запуск остановлен.")

    def on_worker_thread_finished(self):
        self.worker = None
        self.worker_thread = None

        self.start_button.setEnabled(True)
        self.pause_button.setText("Пауза")
        self.pause_button.setEnabled(False)

    def active_elapsed_seconds(self):
        if self.started_at is None:
            return 0.0

        paused_now = 0.0
        if self.is_paused and self.pause_started_at is not None:
            paused_now = time.perf_counter() - self.pause_started_at

        return max(
            0.0,
            time.perf_counter() - self.started_at - self.total_paused_seconds - paused_now,
        )

    def update_elapsed(self):
        if self.started_at is None:
            return

        elapsed = self.active_elapsed_seconds()
        self.elapsed_value.setText(self.format_seconds(elapsed))

        if self.is_paused:
            self.remaining_value.setText("Пауза")
        elif 0.0 < self.last_progress_percent < 100.0:
            # Пересчёт по таймеру учитывает длительный текущий элемент: если
            # прогресс долго не меняется, ETA постепенно увеличивается.
            self.eta_seconds = self.eta_estimator.estimate(
                elapsed, self.last_progress_percent
            )
            if self.eta_seconds is None:
                self.remaining_value.setText("Оценка…")
            else:
                self.remaining_value.setText(self.format_seconds(self.eta_seconds))

    def clear_logs(self):
        self.log_view.clear()

    def log(self, text):
        timestamp = time.strftime("%H:%M:%S")
        self.log_view.append(f"[{timestamp}] {text}")

    @staticmethod
    def format_seconds(seconds):
        seconds = max(0, int(seconds))
        minutes, sec = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)

        if hours:
            return f"{hours} ч {minutes:02d} мин"

        if minutes:
            return f"{minutes} мин {sec:02d} сек"

        return f"{sec} сек"

    def showEvent(self, event):
        super().showEvent(event)
        if self.taskbar_progress is None:
            self.taskbar_progress = WindowsTaskbarProgress(int(self.winId()))

    def closeEvent(self, event):
        if self.worker_thread is not None and self.worker_thread.isRunning():
            if self.worker is not None:
                self.worker.request_stop()

            self.worker_thread.quit()
            self.worker_thread.wait()

        if self.taskbar_progress is not None:
            self.taskbar_progress.clear()
            self.taskbar_progress.close()

        event.accept()