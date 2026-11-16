from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def update_geojson_widths_gpd(
    input_geojson_path,
    width_results,
    output_geojson_path=None,
    width_field="Width",
    source_field=None,
):
    input_geojson_path = Path(input_geojson_path)

    if output_geojson_path is None:
        output_geojson_path = input_geojson_path
    else:
        output_geojson_path = Path(output_geojson_path)

    gdf = gpd.read_file(input_geojson_path).reset_index(drop=True)

    widths_by_feature = {}
    sources_by_feature = {}

    for result in width_results:
        if result.get("width_m") is None:
            continue

        feature_idx = result["edge"]["feature_idx"]
        widths_by_feature.setdefault(feature_idx, []).append(result["width_m"])
        sources_by_feature.setdefault(feature_idx, set()).add(
            result.get("width_source", "mask")
        )

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

        if source_field is not None:
            if source_field not in gdf.columns:
                gdf[source_field] = pd.Series(None, index=gdf.index, dtype="object")
            else:
                gdf[source_field] = gdf[source_field].astype("object")

        for feature_idx, widths in widths_by_feature.items():
            if feature_idx in gdf.index:
                gdf.loc[feature_idx, width_field] = round(
                    float(np.median(widths)),
                    2,
                )
                if source_field is not None:
                    sources = sources_by_feature[feature_idx]
                    gdf.loc[feature_idx, source_field] = (
                        next(iter(sources)) if len(sources) == 1 else "mixed"
                    )

    output_geojson_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_geojson_path, driver="GeoJSON", encoding="utf-8")

    return gdf
