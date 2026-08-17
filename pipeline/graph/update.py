from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def update_geojson_widths_gpd(
    input_geojson_path,
    width_results,
    output_geojson_path=None,
    width_field="Width",
):
    input_geojson_path = Path(input_geojson_path)

    if output_geojson_path is None:
        output_geojson_path = input_geojson_path
    else:
        output_geojson_path = Path(output_geojson_path)

    gdf = gpd.read_file(input_geojson_path).reset_index(drop=True)

    widths_by_feature = {}

    for result in width_results:
        if result.get("width_m") is None:
            continue

        feature_idx = result["edge"]["feature_idx"]
        widths_by_feature.setdefault(feature_idx, []).append(result["width_m"])

    # Если ни для одного ребра ширина не рассчитана, таблицу атрибутов
    # вообще не меняем. Это важно, когда все рёбра исключены из обработки
    # из-за отсутствующих тайлов.
    if widths_by_feature:
        if width_field not in gdf.columns:
            gdf[width_field] = np.nan

        gdf[width_field] = pd.to_numeric(
            gdf[width_field],
            errors="coerce",
        ).astype("float64")

        for feature_idx, widths in widths_by_feature.items():
            if feature_idx in gdf.index:
                gdf.loc[feature_idx, width_field] = round(
                    float(np.median(widths)),
                    2,
                )

    output_geojson_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_geojson_path, driver="GeoJSON", encoding="utf-8")

    return gdf
