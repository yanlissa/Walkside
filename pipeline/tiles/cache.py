from pathlib import Path


def get_cache_image_path(cache_images_dir, x, y, tile_ext):
    return Path(cache_images_dir) / f"{x}_{y}{tile_ext}"


def prepare_tile_cache_images(
    unique_tiles,
    tile_source,
    work_dir,
    tile_ext=".jpg",
    skip_existing=True,
    pause_wait=None,
    stop_requested=None,
    progress_callback=None,
):
    cache_images_dir = Path(work_dir) / "tile_cache" / "images"

    copied = []
    skipped = []
    direct = []
    missing = []
    image_paths = []
    image_items = []
    tiles = list(unique_tiles.values())
    total_tiles = len(tiles)

    if progress_callback is not None:
        progress_callback(0, total_tiles)

    for tile_index, tile in enumerate(tiles, start=1):
        if stop_requested is not None and stop_requested():
            return {
                "cancelled": True,
                "images_dir": cache_images_dir,
                "copied": copied,
                "skipped": skipped,
                "direct": direct,
                "missing": missing,
                "image_paths": image_paths,
                "image_items": image_items,
            }

        if pause_wait is not None and not pause_wait():
            return {
                "cancelled": True,
                "images_dir": cache_images_dir,
                "copied": copied,
                "skipped": skipped,
                "direct": direct,
                "missing": missing,
                "image_paths": image_paths,
                "image_items": image_items,
            }

        destination = get_cache_image_path(
            cache_images_dir,
            tile["x"],
            tile["y"],
            tile_ext,
        )

        action, source = tile_source.save_tile(
            tile,
            destination,
            skip_existing=skip_existing,
        )

        if action == "missing":
            missing.append({
                "tile": {
                    "z": tile["z"],
                    "x": tile["x"],
                    "y": tile["y"],
                },
                "source": source,
            })
        else:
            if action == "direct":
                image_path = Path(source)
                direct.append(image_path)
            else:
                image_path = destination

                if action == "skipped":
                    skipped.append(destination)
                else:
                    copied.append(destination)

            image_paths.append(image_path)
            image_items.append({
                "path": image_path,
                "label_stem": f"{tile['x']}_{tile['y']}",
                "z": tile["z"],
                "x": tile["x"],
                "y": tile["y"],
            })

        if progress_callback is not None:
            progress_callback(tile_index, total_tiles)

    return {
        "cancelled": False,
        "images_dir": cache_images_dir,
        "copied": copied,
        "skipped": skipped,
        "direct": direct,
        "missing": missing,
        "image_paths": image_paths,
        "image_items": image_items,
    }
