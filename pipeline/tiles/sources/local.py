from pathlib import Path

from pipeline.tiles.sources.base import TileSource


class LocalTileSource(TileSource):
    def __init__(self, input_tiles_dir):
        self.input_tiles_dir = Path(input_tiles_dir)

    def get_tile_path(self, z, x, y):
        candidates = []

        for ext in (".jpg", ".jpeg", ".png"):
            # Поддерживаются обе структуры:
            #   root/x_y.jpg
            #   root/z/x/y.jpg
            #   root/z/x_y.jpg
            candidates.append(self.input_tiles_dir / f"{x}_{y}{ext}")
            candidates.append(self.input_tiles_dir / str(z) / str(x) / f"{y}{ext}")
            candidates.append(self.input_tiles_dir / str(z) / f"{x}_{y}{ext}")

        for path in candidates:
            if path.exists():
                return path

        return candidates[0]

    def find_missing(self, unique_tiles, progress_callback=None):
        missing = []
        tiles = list(unique_tiles.values())
        total_tiles = len(tiles)

        if progress_callback is not None:
            progress_callback(0, total_tiles)

        for tile_index, tile in enumerate(tiles, start=1):
            path = self.get_tile_path(tile["z"], tile["x"], tile["y"])

            if not path.exists():
                missing.append({
                    **tile,
                    "expected_path": str(path),
                })

            if progress_callback is not None:
                progress_callback(tile_index, total_tiles)

        return missing

    def save_tile(self, tile, destination, skip_existing=True):
        source = self.get_tile_path(tile["z"], tile["x"], tile["y"])

        if not source.exists():
            return "missing", str(source)

        # Для локального источника исходный тайл используется напрямую.
        # В tile_cache сохраняются только маски, без копии изображения.
        return "direct", str(source)
