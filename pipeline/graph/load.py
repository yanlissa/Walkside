import geopandas as gpd


def load_graph_edges(
    geojson_path,
    target_crs="EPSG:4326",
):
    gdf = gpd.read_file(geojson_path)

    if gdf.crs is None:
        raise ValueError(
            "У входного графа не задан CRS. "
            "Нужно явно задать CRS перед запуском, "
            "например EPSG:4326 или EPSG:3857."
        )

    gdf = gdf.to_crs(target_crs)

    edges = []

    for feature_idx, row in gdf.iterrows():
        geometry = row.geometry

        if geometry is None or geometry.is_empty:
            continue

        properties = row.drop(
            labels="geometry"
        ).to_dict()

        if geometry.geom_type == "LineString":
            coords = list(
                geometry.coords
            )

            if len(coords) >= 2:
                edges.append({
                    "feature_idx": int(feature_idx),
                    "part_idx": 0,
                    "coords": [
                        (float(x), float(y))
                        for x, y, *rest in coords
                    ],
                    "properties": properties,
                })

        elif geometry.geom_type == "MultiLineString":
            for part_idx, line in enumerate(
                geometry.geoms
            ):
                coords = list(
                    line.coords
                )

                if len(coords) >= 2:
                    edges.append({
                        "feature_idx": int(feature_idx),
                        "part_idx": int(part_idx),
                        "coords": [
                            (float(x), float(y))
                            for x, y, *rest in coords
                        ],
                        "properties": properties,
                    })

    return edges