from pathlib import Path

from pipeline.tiles.sources.local import LocalTileSource
from pipeline.tiles.sources.remote import RemoteTileSource


def create_tile_source(source_type, input_tiles_dir=None, work_dir=None):
    if source_type == "local":
        return LocalTileSource(input_tiles_dir)

    if source_type == "remote":
        if work_dir is None:
            raise ValueError("Для remote-источника необходимо передать work_dir")

        cache_db_path = (
            Path(work_dir)
            / "tile_cache"
            / "remote_raw.sqlite3"
        )
        return RemoteTileSource(cache_db_path=cache_db_path)

    raise ValueError(f"Неизвестный источник тайлов: {source_type}")
