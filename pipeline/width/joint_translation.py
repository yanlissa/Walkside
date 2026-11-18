import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares
from shapely.ops import unary_union

from pipeline.width.measurement import iter_lines_from_intersection
from pipeline.width.mosaic import sample_mask
from pipeline.width.skeleton import (
    approximate_axis_lines,
    build_polygon_skeleton_lines,
    estimate_best_global_translation_from_axis,
    split_line_to_segments,
    translate_edge_line,
)

NATIVE_PARAMETERS = dict(
    edge_step=10,
    max_candidate_distance=240,
    max_match_distance=120,
    min_parallel_cos=0.75,
    shift_cluster_radius=12,
    max_candidates=250,
    shift_length_weight=0.001,
)


def stations(line):
    p = np.asarray(line.coords)
    vs = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    s = np.unique(np.r_[np.arange(0, line.length + 1e-8, 10.0), vs, line.length])
    s = s[np.r_[True, np.diff(s) > 1e-7]]
    xy = np.array(
        [[line.interpolate(float(x)).x, line.interpolate(float(x)).y] for x in s]
    )
    tt = []
    for x in s:
        a = line.interpolate(max(0, float(x) - 3))
        b = line.interpolate(min(line.length, float(x) + 3))
        d = np.array([b.x - a.x, b.y - a.y])
        tt.append(d / max(np.linalg.norm(d), 1e-10))
    t = np.array(tt)
    bounds = np.r_[0, (s[:-1] + s[1:]) / 2, line.length]
    w = np.diff(bounds)
    return dict(s=s, xy=xy, t=t, w=w)


JCFG = dict(
    parallel_cos=0.75,
    capture_m=3.0,
    min_direction_support_m=3.0,
    normal_gate_m=1.0,
    sigma_floor_m=0.25,
    sigma_width_fraction=0.1,
    huber_scale=1.0,
    max_seeds=12,
    max_iterations=8,
    convergence_m=0.02,
    ambiguity_loss=0.06,
    distinct_normal_m=1.0,
    rank_ratio=0.05,
    weight_mode="equal",
    min_known_fraction=0.7,
)


def joint_required_support_m(line, cfg):
    return cfg["min_support_fraction"] * line.length * cfg["pixel_size_m"]


def joint_axis_data(pieces):
    axes = []
    owners = []
    polys = {}
    for p in pieces:
        polys[int(p["idx"])] = p["poly"]
        for axis in p["axis_segments"]:
            if axis.length > 1e-7:
                axes.append(axis)
                owners.append(int(p["idx"]))
    if not axes:
        return None
    a = np.array([axis.coords[0] for axis in axes])
    b = np.array([axis.coords[-1] for axis in axes])
    d = b - a
    l2 = np.sum(d * d, axis=1)
    return dict(
        axes=axes,
        owner=np.array(owners),
        poly=polys,
        a=a,
        d=d,
        l2=l2,
        t=d / np.sqrt(l2)[:, None],
    )


def joint_options(st, vector, data, ctx, cfg, k=3):
    p = st["xy"] + vector
    delta = p[:, None, :] - data["a"]
    u = np.clip(np.einsum("ijk,jk->ij", delta, data["d"]) / data["l2"], 0, 1)
    q = data["a"] + u[:, :, None] * data["d"]
    distance = np.linalg.norm(p[:, None, :] - q, axis=2) * cfg["pixel_size_m"]
    cos = np.abs(st["t"] @ data["t"].T)
    score = np.where(cos >= cfg["parallel_cos"], distance, np.inf)
    k = min(k, len(data["axes"]))
    jj = np.argsort(score, axis=1)[:, :k]
    ii = np.arange(len(p))[:, None]
    target = q[ii, jj]
    direction = data["t"][jj].copy()
    direction *= np.where(
        np.sum(direction * st["t"][:, None, :], axis=2) >= 0, 1.0, -1.0
    )[:, :, None]
    normal = np.stack([-direction[:, :, 1], direction[:, :, 0]], axis=2)
    width = np.maximum(
        0.4,
        2 * sample_mask(ctx["dt"], target.reshape(-1, 2), 0.0).reshape(len(p), k) * cfg["pixel_size_m"],
    )
    sigma = np.maximum(cfg["sigma_floor_m"], cfg["sigma_width_fraction"] * width)
    residual = np.sum((p[:, None, :] - target) * normal, axis=2) * cfg["pixel_size_m"]
    valid = (score[ii, jj] <= cfg["capture_m"]) & sample_mask(ctx["known"], p, False)[
        :, None
    ]
    return dict(
        q=target,
        n=normal,
        t=direction,
        width=width,
        sigma=sigma,
        residual=residual,
        distance=distance[ii, jj],
        cos=cos[ii, jj],
        j=jj,
        owner=data["owner"][jj],
        valid=valid,
    )


def joint_seed_vectors(line, st, data, pieces, ctx, cfg):
    candidates = [np.zeros(2)]
    zero = joint_options(
        st, np.zeros(2), data, ctx, {**cfg, "capture_m": cfg["search_m"]}, k=2
    )
    records = []
    for i in np.linspace(0, len(st["s"]) - 1, min(14, len(st["s"]))).astype(int):
        for k in range(zero["j"].shape[1]):
            if (
                zero["distance"][i, k] > cfg["search_m"]
                or zero["cos"][i, k] < cfg["parallel_cos"]
            ):
                continue
            q = zero["q"][i, k]
            n = zero["n"][i, k]
            d = float(n @ (q - st["xy"][i]))
            records.append((n, d))
            candidates.append(q - st["xy"][i])
    for i, (na, da) in enumerate(records):
        for nb, db in records[i + 1 :]:
            A = np.array([na, nb])
            if abs(np.linalg.det(A)) < 0.3:
                continue
            v = np.linalg.solve(A, np.array([da, db]))
            if np.linalg.norm(v) * cfg["pixel_size_m"] <= cfg["max_shift_m"]:
                candidates.append(v)
    for piece in pieces:
        if int(piece["idx"]) not in data["poly"]:
            continue
        seed = estimate_best_global_translation_from_axis(
            line, piece["axis_segments"], translation_pool=cfg["translation_pool"], **NATIVE_PARAMETERS
        )
        if seed is not None:
            candidates.append(np.array([seed["dx"], seed["dy"]]))
    unique = {}
    for v in candidates:
        if np.linalg.norm(v) * cfg["pixel_size_m"] > cfg["max_shift_m"]:
            continue
        key = tuple(np.round(v * cfg["pixel_size_m"] / 0.25).astype(int))
        unique.setdefault(key, np.asarray(v, float))
    ss = {
        k: (v[::3] if isinstance(v, np.ndarray) and len(v) == len(st["s"]) else v)
        for k, v in st.items()
    }
    ranked = []
    for v in unique.values():
        o = joint_options(ss, v, data, ctx, cfg, k=1)
        r = np.abs(o["residual"][:, 0])
        valid = o["valid"][:, 0]
        score = (
            float(np.sum(ss["w"] * cfg["pixel_size_m"] * valid * np.maximum(0, 1 - r / 1.5)))
            - 0.01 * np.linalg.norm(v) * cfg["pixel_size_m"]
        )
        ranked.append((score, v))
    ranked.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, v in ranked:
        if not out or min(np.linalg.norm(v - x) * cfg["pixel_size_m"] for x in out) > 0.4:
            out.append(v)
        if len(out) >= cfg["max_seeds"]:
            break
    if not any(np.linalg.norm(v) < 1e-9 for v in out):
        out.append(np.zeros(2))
    return out


def joint_correspondence_chain(st, vector, data, ctx, cfg):
    o = joint_options(st, vector, data, ctx, cfg, k=3)
    beam = [dict(score=0.0, last=None, path=[])]
    w = st["w"] * cfg["pixel_size_m"]
    for i in range(len(st["s"])):
        successors = []
        for k in range(o["j"].shape[1]):
            if not o["valid"][i, k]:
                continue
            standardized = abs(o["residual"][i, k]) / o["sigma"][i, k]
            reward = w[i] * (1.2 - 0.2 * min(standardized, 3.0) ** 2)
            best = None
            for state in beam:
                penalty = 0.0
                if state["last"] is not None:
                    last, kk = state["last"]
                    dq = (o["q"][i, k] - o["q"][last, kk]) * cfg["pixel_size_m"]
                    dp = (st["xy"][i] - st["xy"][last]) * cfg["pixel_size_m"]
                    consistency = np.linalg.norm(dq - dp)
                    gate = max(
                        1.0, 3 * np.hypot(o["sigma"][i, k], o["sigma"][last, kk])
                    )
                    if consistency > gate or dq @ dp < -0.1:
                        continue
                    penalty = 0.1 * w[i] * (consistency / gate) ** 2
                val = state["score"] + reward - penalty
                if best is None or val > best["score"]:
                    best = dict(score=val, last=(i, k), path=state["path"] + [(i, k)])
            if best is not None:
                successors.append(best)
        # Точки без соответствия можно пропустить.
        alternatives = successors + beam
        alternatives.sort(key=lambda x: x["score"], reverse=True)
        beam = []
        seen = set()
        for state in alternatives:
            key = state["last"]
            if key in seen:
                continue
            beam.append(state)
            seen.add(key)
            if len(beam) >= 8:
                break
        for state in successors:
            if state["last"] not in seen:
                beam.append(state)
                seen.add(state["last"])
        if None not in seen:
            beam.append(dict(score=0.0, last=None, path=[]))
    path = max(beam, key=lambda x: x["score"])["path"] if beam else []
    return o, path


def joint_fit_vector(st, o, path, cfg, vector=None):
    if len(path) < 2:
        return None, 0
    ids = np.array([p[0] for p in path])
    kk = np.array([p[1] for p in path])
    n = o["n"][ids, kk]
    q = o["q"][ids, kk]
    sigma = o["sigma"][ids, kk]
    owner = o["owner"][ids, kk]
    w = st["w"][ids] * cfg["pixel_size_m"]
    poly_ids = np.unique(owner)
    weights = w.copy()
    for pid in poly_ids:
        sel = owner == pid
        L = float(w[sel].sum())
        if cfg["weight_mode"] == "equal":
            total = 1.0
        elif cfg["weight_mode"] == "support":
            total = L
        else:
            total = float(
                np.sum(w[sel] * o["width"][ids[sel], kk[sel]])
            )
        weights[sel] *= total / max(L, 1e-9)
    weights /= weights.sum()
    A = n / sigma[:, None]
    d = np.sum(n * (q - st["xy"][ids]), axis=1) * cfg["pixel_size_m"] / sigma
    H = A.T @ (weights[:, None] * A)
    e, V = np.linalg.eigh(H)
    observed = e >= cfg["rank_ratio"] * max(e.max(), 1e-9)
    direction_support = np.sum(w[:, None] * (n @ V) ** 2, axis=0)
    # Короткий отросток не определяет вторую компоненту сдвига.
    observed &= direction_support >= cfg["min_direction_support_m"]
    B = V[:, observed]
    if not observed.any():
        return None, 0
    offset = np.zeros(2) if vector is None else (vector - B @ (B.T @ vector)) * cfg["pixel_size_m"]

    def residual(z):
        return np.r_[np.sqrt(weights * len(ids)) * (A @ (B @ z + offset) - d), 0.01 * z]

    fit = least_squares(
        residual,
        np.zeros(B.shape[1]),
        loss="huber",
        f_scale=cfg["huber_scale"],
        max_nfev=60,
    )
    return (B @ fit.x + offset) / cfg["pixel_size_m"], int(observed.sum())


def joint_assess(line, st, vector, data, ctx, cfg, o, path):
    if not path:
        return dict(
            feasible=False,
            reasons=["Нет последовательных опор"],
            loss=99.0,
        )
    ids = np.array([x[0] for x in path])
    kk = np.array([x[1] for x in path])
    p = st["xy"][ids] + vector
    owner = o["owner"][ids, kk]
    r = np.abs(o["residual"][ids, kk])
    raw = sample_mask(ctx["raw"], p, False)
    known = sample_mask(ctx["known"], p, False)
    supported = (
        raw
        & known
        & (r <= np.maximum(cfg["normal_gate_m"], 0.25 * o["width"][ids, kk]))
        & (o["distance"][ids, kk] <= np.maximum(1.5, 0.75 * o["width"][ids, kk]))
    )
    support_m = float(np.sum(st["w"][ids[supported]]) * cfg["pixel_size_m"])
    poly_ids = np.unique(owner[supported])
    polys = [data["poly"][int(k)] for k in poly_ids]
    moved = translate_edge_line(line, *vector)
    inside = (
        float(
            moved.intersection(unary_union(polys).buffer(2)).length
            / max(line.length, 1e-9)
        )
        if polys
        else 0.0
    )
    residual = r[supported]
    median = float(np.median(residual)) if len(residual) else None
    known_length = float(np.sum(st["w"][sample_mask(ctx["known"], st["xy"] + vector, False)]) * cfg["pixel_size_m"])
    W = (
        float(np.median(o["width"][ids[supported], kk[supported]]))
        if supported.any()
        else 0.4
    )
    loss = (
        (
            median / max(W, 0.4)
            + 1
            - support_m / max(known_length, 1e-9)
            + 1 - known_length / max(line.length * cfg["pixel_size_m"], 1e-9)
            + 0.01 * np.linalg.norm(vector) * cfg["pixel_size_m"]
        )
        if median is not None
        else 99.0
    )
    reasons = []
    if known_length < cfg["min_known_fraction"] * line.length * cfg["pixel_size_m"]:
        reasons.append("Перенос выводит значительную часть ребра за пределы данных")
    if inside < cfg["inside_min"]:
        reasons.append(f"В маске менее {cfg['inside_min']:.0%} полной длины")
    required_support_m = joint_required_support_m(line, cfg)
    if support_m + 1e-12 < required_support_m:
        reasons.append(
            f'Подтверждено {support_m:.2f} м опор; требуется не менее {100*cfg["min_support_fraction"]:.0f}% полной длины ребра ({required_support_m:.2f} м)'
        )
    if np.linalg.norm(vector) * cfg["pixel_size_m"] > cfg["max_shift_m"]:
        reasons.append("Вектор длиннее радиуса допуска")
    return dict(
        feasible=not reasons,
        reasons=reasons,
        loss=float(loss),
        supported_normals=o["n"][ids[supported], kk[supported]],
    )


def joint_transfer(
    line,
    polygons,
    foreground,
    known,
    pixel_size_m=0.08,
    search_m=8.0,
    max_shift_m=20.0,
    inside_min=0.30,
    min_support_fraction=0.30,
    translation_pool=None,
):
    if line is None or line.is_empty or line.length == 0:
        return dict(vector=(0.0, 0.0), status="not_translated", reason="Пустое ребро")
    cfg = dict(
        JCFG,
        pixel_size_m=pixel_size_m,
        search_m=search_m,
        max_shift_m=max_shift_m,
        inside_min=inside_min,
        min_support_fraction=min_support_fraction,
        translation_pool=translation_pool,
    )
    cfg["min_direction_support_m"] = min(cfg["min_direction_support_m"], joint_required_support_m(line, cfg))
    pieces = []
    for index, polygon in enumerate(polygons):
        if polygon.distance(line) * pixel_size_m > search_m:
            continue
        skeleton = build_polygon_skeleton_lines(polygon)
        axes = [
            segment
            for axis in (approximate_axis_lines(skeleton) or skeleton)
            for inside in iter_lines_from_intersection(axis.intersection(polygon))
            for segment in split_line_to_segments(inside)
            if segment.length > 0
        ]
        if axes:
            pieces.append(dict(idx=index, poly=polygon, axis_segments=axes))
    data = joint_axis_data(pieces)
    if data is None:
        return dict(vector=(0.0, 0.0), status="not_translated", reason="Нет осей в области поиска")

    foreground = foreground.astype(bool) & known
    context = dict(raw=foreground, known=known, dt=distance_transform_edt(foreground))
    samples = stations(line)
    candidates = []
    for seed in joint_seed_vectors(line, samples, data, pieces, context, cfg):
        vector = seed.copy()
        rank = 0
        for _ in range(cfg["max_iterations"]):
            options, path = joint_correspondence_chain(samples, vector, data, context, cfg)
            update, rank = joint_fit_vector(samples, options, path, cfg, vector)
            if update is None or np.linalg.norm(update) * pixel_size_m > max_shift_m:
                break
            change = np.linalg.norm(update - vector) * pixel_size_m
            vector = update
            if change < cfg["convergence_m"]:
                break
        options, path = joint_correspondence_chain(samples, vector, data, context, cfg)
        score = joint_assess(line, samples, vector, data, context, cfg, options, path)
        if not rank and np.linalg.norm(vector) > 1e-6:
            score["feasible"] = False
            score["reasons"].append("Недостаточно направленных опор")
        if all(np.linalg.norm(vector - other) * pixel_size_m >= 0.03 for other, _ in candidates):
            candidates.append((vector, score))

    zero = np.zeros(2)
    options, path = joint_correspondence_chain(samples, zero, data, context, cfg)
    candidates.append((zero, joint_assess(line, samples, zero, data, context, cfg, options, path)))
    candidates.sort(key=lambda item: (not item[1]["feasible"], item[1]["loss"]))
    vector, score = candidates[0]
    if not score["feasible"]:
        return dict(vector=(0.0, 0.0), status="not_translated", reason="; ".join(score["reasons"]))
    normals = score["supported_normals"]
    for other, other_score in candidates[1:]:
        if not other_score["feasible"]:
            break
        difference = (other - vector) * pixel_size_m
        if (
            len(normals)
            and np.quantile(np.abs(normals @ difference), 0.9) >= cfg["distinct_normal_m"]
            and abs(other_score["loss"] - score["loss"]) < cfg["ambiguity_loss"]
        ):
            return dict(vector=(0.0, 0.0), status="not_translated", reason="Неоднозначный сдвиг")
    if np.linalg.norm(vector) <= 1e-6:
        return dict(vector=(0.0, 0.0), status="not_translated", reason="Выбран нулевой вектор")
    return dict(vector=tuple(map(float, vector)), status="translated", reason="Совместный перенос принят")
