import math
from collections import defaultdict, deque


def _edge_key(edge):
    return int(edge["feature_idx"]), int(edge.get("part_idx", 0))


def _positive_width(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def restore_widths_from_neighbors(
    all_edges,
    width_results,
    max_hops=3,
    reference_latitude=55.69,
    width_field="Width",
    preserve_existing=True,
    stop_requested=None,
    pause_wait=None,
    progress_callback=None,
):
    """Метод 4: BFS до max_hops и вес 1 / (steps * (1 + angle)).

    all_edges содержит исходный граф в EPSG:4326, включая промежуточные
    ребра без масок. Восстанавливаются только неудачные измерения из
    width_results. Опоры фиксируются до восстановления; прогнозы не
    становятся новыми опорами. Входные объекты не изменяются.

    Соседство, порядок BFS и углы по концам ребер повторяют ноутбук.
    reference_latitude=55.69 сохранена из исследования намеренно.
    """
    if isinstance(max_hops, bool) or not isinstance(max_hops, int) or max_hops < 1:
        raise ValueError("max_hops должен быть положительным целым числом")
    reference_latitude = float(reference_latitude)
    if not math.isfinite(reference_latitude) or not -90 < reference_latitude < 90:
        raise ValueError("reference_latitude должна лежать между -90 и 90")

    results = [dict(item) for item in width_results]
    stats = {
        "anchors": 0,
        "candidates": 0,
        "restored": 0,
        "unresolved": 0,
        "preserved_existing": 0,
        "skipped_io_errors": 0,
    }

    def finish(cancelled=False):
        return {
            "cancelled": cancelled,
            "width_results": results,
            "max_hops": max_hops,
            **stats,
        }

    def checkpoint():
        if stop_requested is not None and stop_requested():
            return False
        if pause_wait is not None and not pause_wait():
            return False
        return stop_requested is None or not stop_requested()

    if not checkpoint():
        return finish(True)

    edges_by_key = {}
    for index, edge in enumerate(all_edges):
        if index % 256 == 0 and not checkpoint():
            return finish(True)
        key = _edge_key(edge)
        if key in edges_by_key:
            raise ValueError(f"Повторяется ключ ребра: {key}")
        edges_by_key[key] = edge

    donors = {}
    targets = []
    seen_results = set()
    for index, result in enumerate(results):
        if index % 256 == 0 and not checkpoint():
            return finish(True)
        key = _edge_key(result["edge"])
        if key not in edges_by_key:
            raise ValueError(f"Результат относится к ребру вне исходного графа: {key}")
        if key in seen_results:
            raise ValueError(f"Несколько результатов для одной части ребра: {key}")
        seen_results.add(key)
        width = _positive_width(result.get("width_m"))
        is_estimate = (
            result.get("width_source") in ("neighbors", "mixed")
            or result.get("width_mode") == "neighbors"
        )
        if width is not None:
            result["width_m"] = width
            if is_estimate:
                result["width_source"] = result.get("width_source") or "neighbors"
            else:
                result["width_source"] = "mask"
                if result.get("status") == "ok":
                    donors[key] = width
            continue
        result["width_m"] = None
        result["width_px"] = None
        if result.get("status") in ("no_tiles", "missing_images_or_labels"):
            stats["skipped_io_errors"] += 1
            continue
        original_width = edges_by_key[key].get("properties", {}).get(width_field)
        if preserve_existing and _positive_width(original_width) is not None:
            stats["preserved_existing"] += 1
            continue
        targets.append((key, result))

    stats["anchors"] = len(donors)
    stats["candidates"] = len(targets)
    if progress_callback is not None:
        progress_callback(0, len(targets))
    if not targets or not donors:
        stats["unresolved"] = len(targets)
        if progress_callback is not None:
            progress_callback(len(targets), len(targets))
        return finish()

    ends = {}
    node_edges = defaultdict(set)
    sort_keys = {}
    for index, (key, edge) in enumerate(edges_by_key.items()):
        if index % 256 == 0 and not checkpoint():
            return finish(True)
        coords = edge["coords"]
        if len(coords) < 2:
            raise ValueError(f"У ребра недостаточно координат: {key}")
        a = (float(coords[0][0]), float(coords[0][1]))
        b = (float(coords[-1][0]), float(coords[-1][1]))
        if not all(math.isfinite(value) for value in (*a, *b)):
            raise ValueError(f"У ребра некорректные координаты: {key}")
        ends[key] = a, b
        node_edges[a].add(key)
        node_edges[b].add(key)
        sort_keys[key] = (str(edge.get("properties", {}).get("EdgeId", "")), *key)

    adjacency = {}
    angle_cache = {}
    longitude_scale = math.cos(math.radians(reference_latitude))

    def neighbors(key):
        if key not in adjacency:
            a, b = ends[key]
            adjacent = (node_edges[a] | node_edges[b]) - {key}
            adjacency[key] = sorted(adjacent, key=sort_keys.__getitem__)
        return adjacency[key]

    def turning_angle(a, b):
        pair = (a, b) if a < b else (b, a)
        if pair not in angle_cache:
            common = set(ends[a]) & set(ends[b])
            junction = min(common)
            other_a = ends[a][0] if ends[a][1] == junction else ends[a][1]
            other_b = ends[b][0] if ends[b][1] == junction else ends[b][1]
            v = ((junction[0] - other_a[0]) * longitude_scale,
                 junction[1] - other_a[1])
            u = ((other_b[0] - junction[0]) * longitude_scale,
                 other_b[1] - junction[1])
            norm = math.hypot(*v) * math.hypot(*u)
            angle_cache[pair] = (
                math.acos(max(-1.0, min(1.0, (v[0] * u[0] + v[1] * u[1]) / norm)))
                if norm else math.pi
            )
        return angle_cache[pair]

    reachable = None
    if len(donors) * 8 <= len(ends):
        reachable = set(donors)
        frontier = set(donors)
        for _ in range(max_hops):
            next_frontier = set()
            for index, key in enumerate(frontier):
                if index % 256 == 0 and not checkpoint():
                    return finish(True)
                a, b = ends[key]
                next_frontier.update(node_edges[a])
                next_frontier.update(node_edges[b])
            next_frontier.difference_update(reachable)
            if not next_frontier:
                break
            reachable.update(next_frontier)
            frontier = next_frontier

    for index, (target, result) in enumerate(targets):
        if not checkpoint():
            return finish(True)
        if reachable is not None and target not in reachable:
            stats["unresolved"] += 1
            if progress_callback is not None:
                progress_callback(index + 1, len(targets))
            continue
        visited = {target}
        queue = deque([(target, 0, 0.0)])
        support = []
        checked = 0
        while queue:
            current, steps, angle = queue.popleft()
            if steps >= max_hops:
                continue
            for neighbor in neighbors(current):
                checked += 1
                if checked % 256 == 0 and not checkpoint():
                    return finish(True)
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                next_steps = steps + 1
                next_angle = angle + turning_angle(current, neighbor)
                queue.append((neighbor, next_steps, next_angle))
                if neighbor in donors:
                    support.append((neighbor, next_steps, next_angle,
                                    1.0 / (next_steps * (1.0 + next_angle))))

        if support:
            total_weight = sum(item[3] for item in support)
            details = [
                {
                    "feature_idx": key[0],
                    "part_idx": key[1],
                    "edge_id": str(edges_by_key[key].get("properties", {}).get("EdgeId", "")),
                    "width_m": donors[key],
                    "steps": steps,
                    "angle_rad": angle,
                    "weight": weight / total_weight,
                }
                for key, steps, angle, weight in support
            ]
            result.update({
                "mask_status": result.get("status"),
                "status": "ok",
                "width_m": sum(item["width_m"] * item["weight"] for item in details),
                "width_px": None,
                "width_mode": "neighbors",
                "width_source": "neighbors",
                "neighbor_support_count": len(details),
                "neighbor_steps": sorted({item["steps"] for item in details}),
                "neighbor_supports": details,
            })
            stats["restored"] += 1
        else:
            stats["unresolved"] += 1
        if progress_callback is not None:
            progress_callback(index + 1, len(targets))

    return finish()
