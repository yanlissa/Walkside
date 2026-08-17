import math
import os
import threading
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pipeline.tiles.sources.base import TileSource
from pipeline.tiles.sources.sqlite_cache import SQLiteTileCache


YANDEX_SAT_URL = (
    "https://core-sat.maps.yandex.net/tiles"
    "?l=sat&x={x}&y={y}&z={z}&scale=1&lang=ru_RU"
)

YANDEX_E = 0.0818191908426
TILE_SIZE = 256


class RemoteTileSource(TileSource):

    def __init__(
        self,
        cache_db_path,
        timeout_seconds=15,
        connect_timeout_seconds=4,
        retry_total=1,
        retry_backoff_factor=0.3,
        request_delay_seconds=0.0,
        missing_cache_ttl_seconds=24 * 60 * 60,
    ):
        self.timeout_seconds = float(timeout_seconds)
        self.connect_timeout_seconds = float(connect_timeout_seconds)
        self.request_delay_seconds = max(0.0, float(request_delay_seconds))
        self.missing_cache_ttl_seconds = max(
            0.0,
            float(missing_cache_ttl_seconds),
        )

        self.raw_cache = SQLiteTileCache(cache_db_path)

        self.headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": (
                "image/avif,image/webp,image/apng,image/svg+xml,"
                "image/*,*/*;q=0.8"
            ),
        }

        retry = Retry(
            total=int(retry_total),
            connect=int(retry_total),
            read=int(retry_total),
            status=int(retry_total),
            backoff_factor=float(retry_backoff_factor),
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=8,
            pool_maxsize=8,
        )

        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Transient failures must not poison the persistent cache. They are
        # remembered only for this source instance/run so the same unavailable
        # source tile does not wait for another timeout later in the run.
        self._failed_source_tiles = set()

        # Striped locks make cache-check -> HTTP -> cache-write single-flight
        # for the same key if tile preparation becomes parallel later. The
        # fixed number of locks avoids keeping one Lock object per million keys.
        self._source_locks = tuple(threading.Lock() for _ in range(256))

    def find_missing(self, unique_tiles, progress_callback=None):
        # A separate remote existence pass would duplicate traffic. Actual
        # availability is determined while preparing target tiles.
        total_tiles = len(unique_tiles)

        if progress_callback is not None:
            progress_callback(0, total_tiles)
            progress_callback(total_tiles, total_tiles)

        return []

    @staticmethod
    def _build_url(z, x, y):
        return YANDEX_SAT_URL.format(z=z, x=x, y=y)

    def _source_lock(self, key):
        return self._source_locks[hash(key) % len(self._source_locks)]

    @staticmethod
    def _decode_image(content):
        with Image.open(BytesIO(content)) as image:
            image.load()
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def _mark_failed_for_run(self, key, url, reason):
        self._failed_source_tiles.add(key)
        print(
            "SKIP remote source tile "
            f"z={key[0]}, x={key[1]}, y={key[2]}: {reason}; url={url}"
        )

    def _load_source_tile(self, z, x, y):
        """Return one decoded Yandex source tile or None when unavailable."""
        z = int(z)
        x = int(x)
        y = int(y)
        key = (z, x, y)
        url = self._build_url(z, x, y)

        with self._source_lock(key):
            if key in self._failed_source_tiles:
                return None, url

            entry = self.raw_cache.get(
                z,
                x,
                y,
                missing_ttl_seconds=self.missing_cache_ttl_seconds,
            )

            if entry is not None:
                if entry.status == "missing":
                    return None, url

                try:
                    return self._decode_image(entry.content), url
                except (OSError, ValueError, UnidentifiedImageError):
                    # A corrupt cached BLOB is not a real missing tile. Remove
                    # it and immediately try the network in this same run.
                    self.raw_cache.delete(z, x, y)

            if self.request_delay_seconds > 0:
                time.sleep(self.request_delay_seconds)

            try:
                response = self.session.get(
                    url,
                    timeout=(
                        self.connect_timeout_seconds,
                        self.timeout_seconds,
                    ),
                )
            except (
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
            ) as exc:
                self._mark_failed_for_run(
                    key,
                    url,
                    f"{type(exc).__name__}: {exc}",
                )
                return None, url
            except requests.exceptions.RequestException as exc:
                self._mark_failed_for_run(
                    key,
                    url,
                    f"{type(exc).__name__}: {exc}",
                )
                return None, url

            if response.status_code == 404:
                self.raw_cache.put_missing(z, x, y)
                return None, url

            # HTTPAdapter has already exhausted configured retries here.
            if response.status_code == 429 or 500 <= response.status_code <= 599:
                self._mark_failed_for_run(
                    key,
                    url,
                    f"HTTP {response.status_code} after retries",
                )
                return None, url

            try:
                response.raise_for_status()
            except requests.exceptions.RequestException as exc:
                self._mark_failed_for_run(
                    key,
                    url,
                    f"HTTP error: {exc}",
                )
                return None, url

            content_type = response.headers.get("content-type", "")
            if "image" not in content_type.lower():
                self._mark_failed_for_run(
                    key,
                    url,
                    f"invalid content-type={content_type!r}",
                )
                return None, url

            content = response.content

            try:
                image = self._decode_image(content)
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                self._mark_failed_for_run(
                    key,
                    url,
                    f"invalid image: {type(exc).__name__}: {exc}",
                )
                return None, url

            # Store only content that was successfully decoded. UPSERT on the
            # (z, x, y) primary key guarantees one persistent record per tile.
            self.raw_cache.put_ok(z, x, y, content)
            return image, url

    @staticmethod
    def _web_mercator_row_latitudes(z, tile_y):
        """Latitudes at centers of 256 target XYZ/Web Mercator rows."""
        n = 2 ** z
        global_rows = tile_y * TILE_SIZE + np.arange(TILE_SIZE) + 0.5
        world_size = n * TILE_SIZE
        mercator_y = math.pi * (1.0 - 2.0 * global_rows / world_size)
        return np.arctan(np.sinh(mercator_y))

    @staticmethod
    def _latitudes_to_yandex_global_y(latitudes_rad, z):
        """Convert latitudes to global Y pixel coordinates in Yandex Mercator."""
        n = 2 ** z
        sin_phi = np.sin(latitudes_rad)
        mercator = np.log(
            np.tan(math.pi / 4.0 + latitudes_rad / 2.0)
            * (
                (1.0 - YANDEX_E * sin_phi)
                / (1.0 + YANDEX_E * sin_phi)
            ) ** (YANDEX_E / 2.0)
        )
        y_tile = n * (1.0 - mercator / math.pi) / 2.0
        return y_tile * TILE_SIZE

    def _build_reprojected_tile(self, z, x, y):
        latitudes = self._web_mercator_row_latitudes(z, y)
        source_global_y = self._latitudes_to_yandex_global_y(latitudes, z)

        min_source_y = int(math.floor(float(source_global_y.min()) / TILE_SIZE))
        max_source_y = int(math.floor(float(source_global_y.max()) / TILE_SIZE))
        n = 2 ** z

        if x < 0 or x >= n or min_source_y < 0 or max_source_y >= n:
            return None, self._build_url(z, x, min_source_y)

        source_rows = []
        source_urls = []

        for source_y in range(min_source_y, max_source_y + 1):
            image, url = self._load_source_tile(z, x, source_y)
            source_urls.append(url)

            if image is None:
                return None, url

            source_rows.append(image)

        strip = np.concatenate(source_rows, axis=0)

        # Source pixel center 0 is at 0.5 in global pixel coordinates.
        source_index_y = source_global_y - min_source_y * TILE_SIZE - 0.5
        source_index_y = np.clip(source_index_y, 0.0, strip.shape[0] - 1.0)

        y0 = np.floor(source_index_y).astype(np.int64)
        y1 = np.minimum(y0 + 1, strip.shape[0] - 1)
        weight = (source_index_y - y0).astype(np.float32)[:, None, None]

        output = (
            strip[y0].astype(np.float32) * (1.0 - weight)
            + strip[y1].astype(np.float32) * weight
        )
        output = np.clip(np.rint(output), 0, 255).astype(np.uint8)

        image = Image.fromarray(output, mode="RGB")
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=95)

        return buffer.getvalue(), ", ".join(source_urls)

    def save_tile(self, tile, destination, skip_existing=True):
        destination = Path(destination)

        if skip_existing and destination.exists():
            return "skipped", str(destination)

        destination.parent.mkdir(parents=True, exist_ok=True)

        content, source = self._build_reprojected_tile(
            int(tile["z"]),
            int(tile["x"]),
            int(tile["y"]),
        )

        if content is None:
            return "missing", source

        # Target tile stays a normal file because the rest of the application
        # already consumes tile_cache/images paths directly.
        tmp_destination = destination.with_name(
            f"{destination.name}.{os.getpid()}.tmp"
        )
        tmp_destination.write_bytes(content)
        os.replace(tmp_destination, destination)
        return "downloaded", source
