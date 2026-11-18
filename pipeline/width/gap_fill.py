import cv2
import numpy as np
import shapely
from PIL import Image
from shapely.affinity import translate
from shapely.geometry import LineString, MultiLineString, Point, Polygon, box
from shapely.ops import linemerge, unary_union
from shapely.prepared import prep
from skimage.morphology import skeletonize

from pipeline.width.measurement import iter_lines_from_intersection
from pipeline.width.mosaic import sample_mask
from pipeline.width.polygons import polygon_parts
from pipeline.width.skeleton import (
    get_skeleton_neighbors,
    pixel_to_xy,
    trace_skeleton_polyline_from_node,
)


def exact_component_polygon(binary):
    # Центры пикселей имеют целые координаты.
    rectangles = []
    for y in np.flatnonzero(binary.any(axis=1)):
        padded = np.r_[False, binary[y], False].astype(np.int8)
        starts = np.flatnonzero(np.diff(padded) == 1)
        ends = np.flatnonzero(np.diff(padded) == -1)
        rectangles.extend(
            box(float(a) - 0.5, float(y) - 0.5, float(b) - 0.5, float(y) + 0.5)
            for a, b in zip(starts, ends)
        )
    return unary_union(rectangles)


def trace_exact_skeleton(binary, offset):
    skeleton = skeletonize(binary) & binary
    rr, cc = np.where(skeleton)
    pixels = set(zip(rr.tolist(), cc.tolist()))
    nodes = {p for p in pixels if len(get_skeleton_neighbors(p, pixels)) != 2}
    visited = set()
    lines = []
    # Второй обход подбирает замкнутые ветви без узлов.
    for node in sorted(nodes):
        for other in get_skeleton_neighbors(node, pixels):
            if frozenset([node, other]) in visited:
                continue
            path = trace_skeleton_polyline_from_node(
                node, other, pixels, nodes, visited
            )
            if len(path) > 1:
                lines.append(LineString([pixel_to_xy(p, offset) for p in path]))
    for start in sorted(pixels):
        for nxt in get_skeleton_neighbors(start, pixels):
            if frozenset([start, nxt]) in visited:
                continue
            path = trace_skeleton_polyline_from_node(start, nxt, pixels, nodes, visited)
            if len(path) > 1:
                lines.append(LineString([pixel_to_xy(p, offset) for p in path]))
    return lines, skeleton


def raster_inside(line, binary):
    coords = np.rint(np.array(line.coords)).astype(np.int32)
    minx, miny = coords.min(axis=0) - 1
    maxx, maxy = coords.max(axis=0) + 1
    if minx < 0 or miny < 0 or maxx >= binary.shape[1] or maxy >= binary.shape[0]:
        minx = max(0, minx)
        miny = max(0, miny)
        maxx = min(binary.shape[1] - 1, maxx)
        maxy = min(binary.shape[0] - 1, maxy)
    test = np.zeros((maxy - miny + 1, maxx - minx + 1), np.uint8)
    cv2.polylines(
        test,
        [(coords - [minx, miny]).astype(np.int32)],
        False,
        1,
        1,
        lineType=cv2.LINE_8,
    )
    return not np.any((test > 0) & ~binary[miny : maxy + 1, minx : maxx + 1])


def constrained_axis(raw, poly, binary, tolerance=8.0):
    # Хорда должна целиком оставаться в маске.
    prepared = prep(poly)
    result = []

    def visit(points):
        if len(points) < 2:
            return
        chord = LineString([points[0], points[-1]])
        if chord.length < 1e-8:
            mid = len(points) // 2
            if mid > 0:
                visit(points[: mid + 1])
                visit(points[mid:])
            return
        vec = points[-1] - points[0]
        u = np.clip((points - points[0]) @ vec / np.dot(vec, vec), 0, 1)
        deviation = np.linalg.norm(points - (points[0] + u[:, None] * vec), axis=1)
        if (
            max(deviation) <= tolerance
            and prepared.covers(chord)
            and raster_inside(chord, binary)
        ):
            result.append(chord)
            return
        if len(points) == 2:
            if not prepared.covers(chord) or not raster_inside(chord, binary):
                raise ValueError("Скелет выходит за пределы маски")
            result.append(chord)
            return
        k = int(np.argmax(deviation))
        if k == 0 or k == len(points) - 1:
            k = len(points) // 2
        visit(points[: k + 1])
        visit(points[k:])

    for line in raw:
        if line.length >= 3:
            visit(np.array(line.coords, float))
    return result


SIMPLE_CFG = dict(
    step_px=10.0,
    association_m=6.4,
    direction_delta_m=0.24,
    ribbon_step_m=0.08,
    gap_sample_m=0.16,
    gap_min_m=0.8,
    gap_max_m=15.0,
)


def section_frame(line, s_px, delta_px):
    p = np.array(line.interpolate(float(s_px)).coords[0])
    a = np.array(line.interpolate(max(0, s_px - delta_px)).coords[0])
    b = np.array(line.interpolate(min(line.length, s_px + delta_px)).coords[0])
    t = b - a
    t /= max(np.linalg.norm(t), 1e-12)
    return p, t, np.array([-t[1], t[0]])


def constant_runs(flags):
    x = np.r_[False, np.asarray(flags, bool), False].astype(int)
    return list(zip(np.flatnonzero(np.diff(x) == 1), np.flatnonzero(np.diff(x) == -1)))


def skeleton_gap_inventory(line, raw, known, crosswalk, pixel_size_m, config=None):
    cfg = {**SIMPLE_CFG, **(config or {})}
    scale = pixel_size_m
    s = np.unique(
        np.r_[
            np.arange(0, line.length * scale, cfg["gap_sample_m"]), line.length * scale
        ]
    )
    xy = np.array([line.interpolate(float(v / scale)).coords[0] for v in s])
    inside = sample_mask(raw, xy)
    valid = sample_mask(known, xy)
    crossing = sample_mask(crosswalk, xy)
    gaps = []
    excluded = []
    for i, j in constant_runs(~inside):
        a = (s[i - 1] + s[i]) / 2 if i else s[0]
        b = (s[j - 1] + s[j]) / 2 if j < len(s) else s[-1]
        reason = ""
        if i == 0 or j == len(s):
            reason = "Нет наблюдений с двух сторон"
        elif not valid[i:j].all():
            reason = "Нет данных маски"
        elif crossing[i:j].any():
            reason = "Другой класс: переход"
        elif not cfg["gap_min_m"] <= b - a <= cfg["gap_max_m"]:
            reason = "Длина вне диапазона"
        if reason:
            excluded.append(dict(a=a, b=b, reason=reason))
            continue
        left = i - 1
        while left > 0 and inside[left - 1] and valid[left - 1]:
            left -= 1
        right = j
        while right + 1 < len(s) and inside[right + 1] and valid[right + 1]:
            right += 1
        gaps.append(
            dict(
                a=a,
                b=b,
                left_interval=(s[left], s[i - 1]),
                right_interval=(s[j], s[right]),
                ordinal=len(gaps) + 1,
            )
        )
    return gaps, excluded


def median_q15_65(values):
    w = np.asarray(values, float)
    if not len(w):
        return np.nan
    lo, hi = np.quantile(w, [0.15, 0.65])
    inside = w[(w >= lo) & (w <= hi)]
    return float(np.median(inside if len(inside) else w))


def group_length_weights(group):
    s = np.array([v["s"] for v in group])
    ds = np.diff(s)
    weights = (
        np.r_[ds[0] / 2, (ds[:-1] + ds[1:]) / 2, ds[-1] / 2]
        if len(s) > 1
        else np.ones(1)
    )
    weights = np.maximum(weights, 1e-9)
    return weights / weights.sum()


def weighted_median(values, weights):
    order = np.argsort(values)
    v = np.asarray(values)[order]
    w = np.asarray(weights)[order]
    return float(v[np.searchsorted(np.cumsum(w), 0.5 * w.sum(), side="left")])


def group_reference(group):
    weights = group_length_weights(group)
    return dict(
        s=weighted_median([v["s"] for v in group], weights),
        width=weighted_median([v["width"] for v in group], weights),
        offset_t=weighted_median([v["offset_t"] for v in group], weights),
        offset_n=weighted_median([v["offset_n"] for v in group], weights),
        angle=weighted_median(
            np.unwrap(2 * np.array([v["angle"] for v in group])) / 2, weights
        ),
    )


def restore_single_gap(
    line,
    sections,
    gap,
    raw,
    known,
    crosswalk,
    pixel_size_m,
    config=None,
    anchor_selection=None,
):
    cfg = {**SIMPLE_CFG, **(config or {})}
    scale = pixel_size_m
    groups, audit = (
        anchor_selection
        if anchor_selection is not None
        else single_anchors(sections, gap, scale, cfg)
    )
    row = dict(
        status="нет якорей",
        reason=audit["reason"],
        connected=False,
        added_pixels=0,
        left_N=audit["sides"]["left"]["N"],
        left_k=audit["sides"]["left"]["k"],
        right_N=audit["sides"]["right"]["N"],
        right_k=audit["sides"]["right"]["k"],
    )
    result = dict(groups=groups, audit=audit, pred=None)
    if groups is None:
        return row, result
    refs = [group_reference(g) for g in groups]
    a, b = [v["s"] for v in refs]
    query = np.unique(
        np.r_[
            np.linspace(a, b, max(2, int(np.ceil((b - a) / cfg["ribbon_step_m"])) + 1)),
            gap["a"],
            gap["b"],
        ]
    )
    widths, info = single_width_model(sections, groups, query)
    row.update(info)
    if widths is None:
        row["status"] = "модель отклонена"
        return row, result
    xy = []
    normals = []
    for s in query:
        p, t, n = section_frame(line, s / scale, cfg["direction_delta_m"] / scale)
        tau = (s - a) / (b - a)
        dt = (1 - tau) * refs[0]["offset_t"] + tau * refs[1]["offset_t"]
        dn = (1 - tau) * refs[0]["offset_n"] + tau * refs[1]["offset_n"]
        if s < gap["a"]:
            angle = refs[0]["angle"] * (gap["a"] - s) / max(gap["a"] - a, 1e-9)
        elif s > gap["b"]:
            angle = refs[1]["angle"] * (s - gap["b"]) / max(b - gap["b"], 1e-9)
        else:
            angle = 0.0
        xy.append(p + (dt * t + dn * n) / scale)
        normals.append(np.sin(angle) * t + np.cos(angle) * n)
    xy = np.asarray(xy)
    normals = np.asarray(normals)
    lower = xy - widths[:, None] / (2 * scale) * normals
    upper = xy + widths[:, None] / (2 * scale) * normals
    ring = Polygon(np.r_[lower, upper[::-1]])
    if not ring.is_valid or ring.area <= 0:
        row.update(
            status="геометрия отклонена",
            reason="Самопересечение или вырождение границ полосы",
        )
        result.update(query=query, widths=widths)
        return row, result
    signed = []
    for i in range(len(query) - 1):
        quad = np.array([lower[i], lower[i + 1], upper[i + 1], upper[i]])
        signed.append(
            float(
                np.sum(
                    quad[:, 0] * np.roll(quad[:, 1], -1)
                    - quad[:, 1] * np.roll(quad[:, 0], -1)
                )
            )
        )
    if np.any(np.asarray(signed) <= 1e-9):
        row.update(
            status="геометрия отклонена",
            reason="Перекручивание последовательных сечений",
        )
        return row, result
    pred = np.zeros(raw.shape, np.uint8)
    for i in range(len(query) - 1):
        cv2.fillPoly(
            pred,
            [
                np.rint([lower[i], lower[i + 1], upper[i + 1], upper[i]]).astype(
                    np.int32
                )
            ],
            1,
        )
    pred = (pred > 0) & known & ~raw & ~crosswalk
    yy, xx = np.where(pred)
    if len(xx):
        projected = shapely.line_locate_point(line, shapely.points(xx, yy)) * scale
        allowed = (projected >= gap["a"] - scale) & (projected <= gap["b"] + scale)
        pred[yy[~allowed], xx[~allowed]] = False
    region = np.zeros(raw.shape, np.uint8)
    pad = max(
        1,
        int(
            np.ceil(max(widths.max(), *[v["width"] for g in groups for v in g]) / scale)
        ),
    )
    for p, q in zip(xy[:-1], xy[1:]):
        cv2.line(
            region, tuple(np.rint(p).astype(int)), tuple(np.rint(q).astype(int)), 1, pad
        )
    labels = cv2.connectedComponents(((raw | pred) & (region > 0)).astype(np.uint8), 8)[
        1
    ]
    endpoints = np.array(
        [
            np.mean(min(g, key=lambda v: abs(v["s"] - r["s"]))["ends"], axis=0)
            for g, r in zip(groups, refs)
        ]
    )
    hits = sample_mask(labels, endpoints)
    connected = bool(hits[0] > 0 and hits[0] == hits[1] and pred.any())
    row.update(
        status="стык подтверждён" if connected else "нет двустороннего стыка",
        reason="" if connected else "Нет локального двустороннего пиксельного стыка",
        connected=connected,
        added_pixels=int(pred.sum()),
        predicted_width_min_m=float(widths.min()),
        predicted_width_max_m=float(widths.max()),
    )
    result.update(pred=pred, query=query, widths=widths, centerline=xy, normals=normals)
    return row, result


def associate_initial_sections(measured, line, pixel_size_m, config=None):
    cfg = {**SIMPLE_CFG, **(config or {})}
    scale = pixel_size_m
    targets = {}
    records = []
    for number, section in enumerate(measured):
        center = np.asarray(section["center"])
        distance = float(Point(center).distance(line) * scale)
        if distance > cfg["association_m"]:
            continue
        s_px = float(line.project(Point(center)))
        p, t, n = section_frame(line, s_px, 3.0)
        if abs(np.asarray(section["axis_tangent"]) @ t) < 0.75 or LineString(section["ends"]).distance(line) > 1.0:
            continue
        ends = np.asarray(section["ends"])
        if (ends[1] - ends[0]) @ n < 0:
            ends = ends[::-1]
        chord_normal = (ends[1] - ends[0]) / section["width_px"]
        middle = ends.mean(axis=0)
        offset = (middle - p) * scale
        targets.setdefault(section["owner"], []).append(section["width_px"] * scale)
        records.append(
            dict(
                section,
                s=s_px * scale,
                width=section["width_px"] * scale,
                ends=ends.tolist(),
                section_id=number,
                offset_t=float(offset @ t),
                offset_n=float(offset @ n),
                angle=float(np.arctan2(chord_normal @ t, chord_normal @ n)),
                center_distance_m=distance,
            )
        )
    targets = {owner: median_q15_65(values) for owner, values in targets.items()}
    for record in records:
        record["piece_target_width_m"] = targets[record["owner"]]
    return sorted(records, key=lambda v: (v["s"], v["owner"], v["section_id"]))


def single_anchors(sections, gap, pixel_size_m, config=None):
    groups = []
    audit = {}
    for side, bound in [("left", gap["a"]), ("right", gap["b"])]:
        owner = gap.get(side + "_owner")
        interval = gap[side + "_interval"]
        candidates = [
            v
            for v in sections
            if v["owner"] == owner and interval[0] <= v["s"] <= interval[1]
            and (v["s"] <= bound if side == "left" else v["s"] >= bound)
        ]
        if not candidates:
            audit[side] = dict(
                N=0,
                k=0,
                reason="Нет реального скелетного сечения своего фрагмента на стороне разрыва",
            )
            groups.append(None)
            continue
        target = float(candidates[0]["piece_target_width_m"])
        if not np.isfinite(target):
            audit[side] = dict(
                N=len(candidates),
                k=0,
                reason="Нет измеренной характерной ширины фрагмента",
            )
            groups.append(None)
            continue
        best_difference = min(abs(v["width"] - target) for v in candidates)
        tolerance = max(2 * pixel_size_m, 0.08 * target)
        near_median = [
            v
            for v in candidates
            if abs(v["width"] - target) <= best_difference + tolerance
        ]
        selected = dict(min(near_median, key=lambda v: abs(v["s"] - bound)))
        selected.update(fit_side=side, fit_bound=bound, fit_interval=interval)
        nearest = min(candidates, key=lambda v: abs(v["s"] - bound))
        audit[side] = dict(
            N=len(candidates),
            k=1,
            owner=owner,
            section_id=selected["section_id"],
            target_width_m=target,
            anchor_width_m=selected["width"],
            near_end_width_m=nearest["width"],
            distance_from_gap_m=abs(selected["s"] - bound),
            target_difference_m=abs(selected["width"] - target),
            target_tolerance_m=tolerance,
            best_difference_m=best_difference,
            reason="",
        )
        groups.append([selected])
    reason = "; ".join(
        side + ": " + audit[side]["reason"]
        for side in ["left", "right"]
        if audit[side]["reason"]
    )
    return None if any(g is None for g in groups) else groups, dict(
        sides=audit, reason=reason
    )


def observations_for_anchor(sections, anchor):
    return sorted(
        [
            v
            for v in sections
            if v["owner"] == anchor["owner"] and anchor["fit_interval"][0] <= v["s"] <= anchor["fit_interval"][1]
            and (
                v["s"] <= anchor["fit_bound"]
                if anchor["fit_side"] == "left"
                else v["s"] >= anchor["fit_bound"]
            )
        ],
        key=lambda v: v["s"],
    )


def single_width_model(sections, groups, query):
    left, right = groups[0][0], groups[1][0]
    a = left["s"]
    b = right["s"]
    if b - a <= 1e-9:
        return None, dict(reason="Не определён порядок двух реальных якорей")
    sides = [observations_for_anchor(sections, v) for v in [left, right]]
    if any(not side for side in sides):
        return None, dict(reason="Нет исходных измерений для модели ширины")
    obs = sides[0] + sides[1]
    x = (np.array([v["s"] for v in obs]) - a) / (b - a)
    y = np.array([v["width"] for v in obs])
    A = np.vander(x, 3, increasing=True)
    weights = np.r_[group_length_weights(sides[0]), group_length_weights(sides[1])] * 0.5
    if np.linalg.matrix_rank(A) < 3:
        return None, dict(reason="Недостаточно независимых исходных сечений для полинома второй степени")
    coef = np.linalg.lstsq(A * np.sqrt(weights[:, None]), y * np.sqrt(weights), rcond=None)[0]
    q = (np.asarray(query) - a) / (b - a)
    pred = np.polynomial.polynomial.polyval(q, coef)
    if abs(coef[2]) > 1e-12:
        vertex = -coef[1] / (2 * coef[2])
        if q.min() <= vertex <= q.max() and np.polynomial.polynomial.polyval(vertex, coef) <= 0:
            return None, dict(reason="Неположительная ширина внутри разрыва у полинома")
    if not np.isfinite(pred).all() or np.any(pred <= 0):
        return None, dict(
            reason="Нечисловая или неположительная ширина; прогноз не обрезается"
        )
    return pred, dict(reason="")


def whole_mask_axes_and_sections(raw, known, step_px=10.0):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw.astype(np.uint8), 8)
    pieces = []
    records = []
    all_axes = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < 80:
            continue
        component = labels[y : y + h, x : x + w] == label
        polygon = translate(
            exact_component_polygon(component), xoff=float(x), yoff=float(y)
        )
        paths, _ = trace_exact_skeleton(component, (int(x), int(y)))
        axes = constrained_axis(paths, polygon, raw)
        for j, poly in enumerate(polygon_parts(polygon)):
            local = [axis for axis in axes if poly.covers(axis)]
            if not local:
                continue
            owner = label * 1000 + j
            piece = dict(idx=owner, poly=poly, axis_segments=local)
            pieces.append(piece)
            all_axes.extend(local)
            merged = iter_lines_from_intersection(
                linemerge(MultiLineString([list(axis.coords) for axis in local]))
            )
            bx = poly.bounds
            radius = np.hypot(bx[2] - bx[0], bx[3] - bx[1]) + 2
            for branch, path in enumerate(merged):
                for number, d in enumerate(
                    np.arange(0, float(path.length) + 1e-9, step_px)
                ):
                    center, tangent, normal = section_frame(path, float(d), 3.0)
                    if (
                        not sample_mask(raw, center[None, :])[0]
                        or not sample_mask(known, center[None, :])[0]
                    ):
                        continue
                    ray = LineString(
                        [center - radius * normal, center + radius * normal]
                    )
                    cuts = [
                        v
                        for v in iter_lines_from_intersection(poly.intersection(ray))
                        if v.length > 1e-7 and v.distance(Point(center)) < 1e-7
                    ]
                    if not cuts:
                        continue
                    chord = cuts[0]
                    records.append(
                        dict(
                            owner=owner,
                            branch=f"{owner}:{branch}",
                            sample_index=number,
                            path_s_px=float(d),
                            center=center.tolist(),
                            ends=[list(chord.coords[0]), list(chord.coords[-1])],
                            width_px=float(chord.length),
                            axis_tangent=tangent.tolist(),
                            normal=normal.tolist(),
                            source="raster_skeleton_section_10px",
                            step_px=float(step_px),
                        )
                    )
    return pieces, all_axes, records


def restore_mask_gaps(mask_mosaic, line, known, pixel_size_m=0.08):
    rgb = np.array(mask_mosaic.convert("RGB"))
    raw = np.all(rgb == [61, 61, 245], axis=2) & known
    crosswalk = np.all(rgb == [255, 4, 4], axis=2)
    gaps, _ = skeleton_gap_inventory(line, raw, known, crosswalk, pixel_size_m)
    if not gaps:
        return mask_mosaic, []
    pieces, _, initial = whole_mask_axes_and_sections(raw, known)
    sections = associate_initial_sections(initial, line, pixel_size_m)
    additions = np.zeros_like(raw)
    report = []
    for gap in gaps:
        for side, position in (("left", gap["left_interval"][1]), ("right", gap["right_interval"][0])):
            point = Point(np.floor(np.asarray(line.interpolate(position / pixel_size_m).coords[0]) + 0.5))
            nearby = min(pieces, key=lambda piece: piece["poly"].distance(point), default=None)
            gap[side + "_owner"] = (
                nearby["idx"] if nearby is not None and nearby["poly"].distance(point) <= 1.0 else None
            )
        row, result = restore_single_gap(line, sections, gap, raw, known, crosswalk, pixel_size_m)
        report.append(row)
        if row["connected"]:
            additions |= result["pred"]
    if not additions.any():
        return mask_mosaic, report
    rgb[additions & known & ~raw & ~crosswalk] = [61, 61, 245]
    return Image.fromarray(rgb), report
