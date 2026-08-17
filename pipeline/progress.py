from __future__ import annotations

from collections import deque
from dataclasses import dataclass


# Доли этапов в общем прогрессе. Самые тяжёлые этапы получают больший вес.
STAGE_RANGES = {
    "load_graph": (0.0, 3.0),
    "tile_indices": (3.0, 10.0),
    "find_tiles": (10.0, 13.0),
    "prepare_tiles": (13.0, 22.0),
    "predict_masks": (22.0, 68.0),
    "calculate_widths": (68.0, 98.0),
    "update_graph": (98.0, 100.0),
}


@dataclass(frozen=True)
class ProgressInfo:
    stage_key: str
    stage_title: str
    current: int
    total: int
    stage_percent: float
    overall_percent: float
    message: str = ""

    def as_dict(self) -> dict:
        return {
            "stage_key": self.stage_key,
            "stage_title": self.stage_title,
            "current": self.current,
            "total": self.total,
            "stage_percent": self.stage_percent,
            "overall_percent": self.overall_percent,
            "message": self.message,
        }


def build_progress(
    stage,
    current=0,
    total=1,
    message="",
    fraction_override: float | None = None,
) -> ProgressInfo:
    start, end = STAGE_RANGES[stage.key]
    original_total = max(0, int(total))
    safe_total = max(1, original_total)
    current = min(max(0, int(current)), safe_total)
    fraction = (
        min(max(float(fraction_override), 0.0), 1.0)
        if fraction_override is not None
        else current / safe_total
    )

    return ProgressInfo(
        stage_key=stage.key,
        stage_title=stage.title,
        current=min(current, original_total) if original_total else 0,
        total=original_total,
        stage_percent=fraction * 100.0,
        overall_percent=start + (end - start) * fraction,
        message=message,
    )


class EtaEstimator:
    def __init__(self, max_samples: int = 120, recent_window_seconds: float = 90.0):
        self.samples = deque(maxlen=max_samples)
        self.recent_window_seconds = float(recent_window_seconds)
        self.smoothed_eta: float | None = None

    def reset(self) -> None:
        self.samples.clear()
        self.smoothed_eta = None

    def add_sample(self, elapsed_seconds: float, percent: float) -> None:
        elapsed = max(0.0, float(elapsed_seconds))
        percent = min(max(0.0, float(percent)), 100.0)

        if self.samples:
            last_elapsed, last_percent = self.samples[-1]
            if elapsed < last_elapsed:
                return
            if percent <= last_percent + 1e-6:
                return

        self.samples.append((elapsed, percent))

    def estimate(self, elapsed_seconds: float, percent: float) -> float | None:
        elapsed = max(0.0, float(elapsed_seconds))
        percent = min(max(0.0, float(percent)), 100.0)

        if percent >= 100.0:
            self.smoothed_eta = 0.0
            return 0.0

        if elapsed < 4.0 or percent < 0.5 or len(self.samples) < 2:
            return None

        # Средняя скорость с начала запуска.
        global_rate = percent / max(elapsed, 1e-9)

        newest_time = elapsed
        newest_percent = percent
        oldest_time, oldest_percent = self.samples[0]

        for sample_time, sample_percent in self.samples:
            if newest_time - sample_time <= self.recent_window_seconds:
                oldest_time, oldest_percent = sample_time, sample_percent
                break

        delta_time = newest_time - oldest_time
        delta_percent = newest_percent - oldest_percent
        recent_rate = (
            delta_percent / delta_time
            if delta_time >= 2.0 and delta_percent >= 0.05
            else 0.0
        )

        if recent_rate > 0.0:
            rate = 0.7 * recent_rate + 0.3 * global_rate
        else:
            rate = global_rate

        if rate <= 1e-9:
            return None

        raw_eta = (100.0 - percent) / rate
        raw_eta = min(max(raw_eta, 0.0), 7 * 24 * 3600.0)

        if self.smoothed_eta is None:
            self.smoothed_eta = raw_eta
        else:
            # Сильнее реагируем на рост ETA (замедление), мягче — на падение.
            alpha = 0.35 if raw_eta > self.smoothed_eta else 0.18
            self.smoothed_eta = alpha * raw_eta + (1.0 - alpha) * self.smoothed_eta

        return self.smoothed_eta
