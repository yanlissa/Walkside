from abc import ABC, abstractmethod


class TileSource(ABC):
    @abstractmethod
    def find_missing(self, unique_tiles, progress_callback=None):
        pass

    @abstractmethod
    def save_tile(self, tile, destination, skip_existing=True):
        pass
