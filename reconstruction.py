"""Data preparation, hourly reconstruction and validation for Narsjø."""

from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve

folder = Path(__file__).resolve().parent
raw_folder = folder / "data" / "raw"
saved_folder = folder / "data" / "processed"
reference_folder = folder / "data" / "reference"
output_folder = folder / "results"
target_station = "2.11.0"
donor_stations = {"2.265.0": "Unsetåa", "1083": "Lillglän", "1223": "Tänndalen"}
reference_years = [1994, 1995, 1996, 2000, *range(2015, 2026)]
excluded_years = {1993, 1999, 2007}
excluded_donor_years = {2002, 2007, 2011, 2013, 2015}
boundary_dates = ("1999-01-01T00:00Z", "2007-01-01T00:00Z")
methods = ("linear", "daily_linear", "donor_ols", "daily_donor_ols")
lag_candidates = (0, -6, 6, -12, 12, -24, 24)
error_limits = (0.1, 0.1, 0.15)
minimum_tests, minimum_years, minimum_events = (12, 4, 10)
maximum_gap = 1050
maximum_year_gap = 120
separation_hours = 312
bootstrap_samples = 1000
random_seed = 4300
return_periods = (20, 200, 1000)


def read_nve(name):
    """Read the original values and provider flags without changing them."""
    frame = pd.read_csv(raw_folder / name, sep=";", decimal=",", skiprows=1)
    frame.index = pd.to_datetime(frame.pop("Tidspunkt"), utc=True)
    assert frame.index.is_unique and frame.index.is_monotonic_increasing
    return frame


def read_smhi(name):
    """Average four approved quarter-hour readings into each UTC hour."""
    quarter = pd.read_csv(
        raw_folder / name, sep=";", skiprows=6, usecols=[0, 1, 2], encoding="utf-8-sig"
    )
    quarter.index = pd.to_datetime(quarter.pop("Datum Tid (UTC)"), utc=True)
    assert quarter.index.is_unique and quarter.index.is_monotonic_increasing
    assert quarter.index.minute.isin([0, 15, 30, 45]).all()
    values = pd.to_numeric(quarter["Vattenföring (15 min)"], errors="raise")
    approved = values.ge(0) & np.isfinite(values) & quarter.Kvalitet.eq("G")
    hourly = values.where(approved).resample("h")
    return hourly.mean().where(hourly.count().eq(4))


def read_inputs():
    """Align the three donor series while preserving missing hours."""
    files = pd.read_csv(folder / "data/sources.csv", dtype={"station_id": str}).to_dict(
        "records"
    )
    aligned, corrected = ({}, {})
    for item in files:
        station = item["station_id"]
        if station not in donor_stations:
            continue
        if item["provider"] == "SMHI":
            corrected[station] = read_smhi(item["file"])
        else:
            frame = read_nve(item["file"])
            values = frame["Vannføring (m³/s)"]
            accepted = (
                values.ge(0)
                & np.isfinite(values)
                & frame.Kontrollert.eq(3)
                & frame.Korrigert.isin([0, 1])
                & ~frame.index.year.isin(excluded_donor_years)
            )
            aligned[station] = frame
            corrected[station] = values.where(accepted)
    assert set(corrected) == set(donor_stations)
    return {"corrected": pd.DataFrame(corrected), "aligned": aligned}


def daily_source():
    frame = pd.read_csv(
        raw_folder / "2.11.0-Vannføring-dogn-v0.csv",
        sep=";",
        decimal=",",
        skiprows=1,
        usecols=[0, 1, 2, 3],
    )
    dates = pd.to_datetime(frame["Tidspunkt"], utc=True, errors="coerce")
    frame.index = dates
    frame = frame.loc[frame.index.notna()].copy()
    values = pd.to_numeric(
        frame["Vannføring (m³/s)"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    assert (frame.index.hour == 11).all()
    for stamp, value, correction, control in zip(
        frame.index, values, frame.Korrigert, frame.Kontrollert
    ):
        if np.isfinite(value) and value >= 0:
            yield (
                stamp - pd.Timedelta(hours=12),
                value,
                str(int(correction)),
                str(int(control)),
            )


def read_reconstruction(folder=saved_folder):
    """Load the verified hourly grid, year decisions and validation scores."""

    def read(name):
        path = folder / name
        return pd.read_csv(path, low_memory=False)

    hourly = read("hourly_reconstruction.csv")
    hourly.index = pd.to_datetime(hourly.timestamp_utc, utc=True)
    assert hourly.index.is_unique
    assert (hourly.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
    return (hourly, read("year_summary.csv"), read("validation_scores.csv"))


def runs(mask):
    changes = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def extract(values, threshold, separation=separation_hours, valley_rule=True):
    """Unknown hours are barriers, never evidence of a below-threshold interval.

    Inputs use an uninterrupted hourly grid. Equal-to-threshold samples do not
    exceed. A gap of exactly `separation` between exceedances stays in one
    cluster. Independent adjacent clusters must also have a valley strictly
    below two-thirds of the first chronological peak.
    """
    q = np.asarray(values, float)
    result = []
    for left, right in runs(np.isfinite(q)):
        above = np.flatnonzero(q[left:right] > threshold) + left
        if not len(above):
            continue
        blocks = np.split(above, np.flatnonzero(np.diff(above) > separation) + 1)
        groups = [[int(x[0]), int(x[-1]), int(x[np.argmax(q[x])])] for x in blocks]
        i = 0
        while valley_rule and i < len(groups) - 1:
            first, second = groups[i : i + 2]
            valley = np.min(q[first[2] : second[2] + 1])
            if valley >= 2 / 3 * q[first[2]]:
                peak = first[2] if q[first[2]] >= q[second[2]] else second[2]
                groups[i : i + 2] = [[first[0], second[1], peak]]
                i = max(0, i - 1)
            else:
                i += 1
        for first, last, peak in groups:
            spaced = first - left >= separation and right - 1 - last >= separation
            complete = (
                spaced
                and np.min(q[left:first]) < 2 / 3 * threshold
                and (np.min(q[last + 1 : right]) < 2 / 3 * q[peak])
            )
            result.append(
                {
                    "first": first,
                    "last": last,
                    "peak": peak,
                    "left": int(left),
                    "right": int(right),
                    "complete": bool(complete),
                }
            )
    return result


def context_bounds(q, first, stop, threshold, separation=separation_hours):
    """Extend until both edges have a full quiet run; return None at unknowns.

    The caller supplies the whole series (and substitutes a prediction when
    appropriate), so a flood extending outside an initial 312h margin is not
    silently truncated.
    """
    left, right = (max(0, first - separation), min(len(q), stop + separation))
    while True:
        if not np.isfinite(q[left:right]).all():
            return None
        left_ok = left <= first - separation and np.all(
            q[left : left + separation] <= threshold
        )
        right_ok = right >= stop + separation and np.all(
            q[right - separation : right] <= threshold
        )
        if left_ok and right_ok:
            break
        new_left = left if left_ok else max(0, left - separation)
        new_right = right if right_ok else min(len(q), right + separation)
        if (new_left, new_right) == (left, right):
            return None
        left, right = (new_left, new_right)
    unknown = np.flatnonzero(~np.isfinite(q))
    before = unknown[unknown < first]
    after = unknown[unknown >= stop]
    finite_left = int(before[-1] + 1) if len(before) else 0
    finite_right = int(after[0]) if len(after) else len(q)
    for event in extract(q[finite_left:finite_right], threshold, separation):
        a, b, p = (
            finite_left + event["first"],
            finite_left + event["last"],
            finite_left + event["peak"],
        )
        if b + separation < first or a - separation >= stop:
            continue
        if not event["complete"]:
            return None
        low_left = np.flatnonzero(q[finite_left:a] < 2 / 3 * threshold) + finite_left
        low_right = np.flatnonzero(q[b + 1 : finite_right] < 2 / 3 * q[p]) + b + 1
        left = min(left, a - separation, int(low_left[-1]))
        right = max(right, b + separation + 1, int(low_right[0] + 1))
    return (int(left), int(right))


def affected(events, positions, separation=separation_hours):
    return [
        e
        for e in events
        if e["complete"]
        and np.any(
            (positions >= e["first"] - separation)
            & (positions <= e["last"] + separation)
        )
    ]


def decision_sensitive(values, positions, blocks, peak_upper, threshold):
    """Screen decision relevance inside an empirically supported peak envelope.

    Proximity to a flood prompts testing, but does not itself establish that a
    low-flow fill can change the event selection. A finite upper envelope must
    come from sufficient validation, never from a low point prediction alone.
    This is a conditional risk screen, not a deterministic bound on real flow.
    """
    if not np.isfinite(peak_upper) or peak_upper >= threshold:
        return (True, "possible_hidden_threshold_exceedance")
    q = np.asarray(values, float)
    edges = [i for a, b in blocks for i in (a - 1, b)]
    if all(
        (
            0 <= i < len(q) and np.isfinite(q[i]) and (q[i] < 2 / 3 * threshold)
            for i in edges
        )
    ):
        return (False, "below_threshold_envelope_and_observed_low_boundaries")
    lower, upper = (q.copy(), q.copy())
    lower[positions], upper[positions] = (0.0, peak_upper)
    bounds = context_bounds(upper, int(positions[0]), int(positions[-1] + 1), threshold)
    if bounds is None:
        return (True, "unresolved_context_or_valley")
    left, right = bounds
    initial = extract(upper[left:right], threshold, valley_rule=False)
    peaks = [e["peak"] + left for e in initial]
    for i, first in enumerate(peaks):
        cutoff = 2 / 3 * q[first]
        for second in peaks[i + 1 :]:
            low_pass = np.min(lower[first : second + 1]) < cutoff
            high_pass = np.min(upper[first : second + 1]) < cutoff
            if low_pass != high_pass:
                return (True, "possible_change_to_between_peak_valley")
    return (False, "event_selection_stable_within_peak_envelope")


def compare(truth, estimate, changed, threshold, origin=0):
    """Match overlapping events one-to-one; merging/splitting is penalised."""
    observed_events, predicted_events = (
        extract(truth, threshold),
        extract(estimate, threshold),
    )
    near = lambda e: np.any(
        (changed >= e["first"] - separation_hours)
        & (changed <= e["last"] + separation_hours)
    )
    actual_ids = {i for i, e in enumerate(observed_events) if near(e)}
    predicted_ids = {i for i, e in enumerate(predicted_events) if near(e)}
    for _ in range(len(observed_events) + len(predicted_events) + 1):
        old = (len(actual_ids), len(predicted_ids))
        actual_ids.update(
            (
                i
                for i, e in enumerate(observed_events)
                if any(
                    (
                        max(e["first"], predicted_events[j]["first"])
                        <= min(e["last"], predicted_events[j]["last"])
                        for j in predicted_ids
                    )
                )
            )
        )
        predicted_ids.update(
            (
                j
                for j, e in enumerate(predicted_events)
                if any(
                    (
                        max(e["first"], observed_events[i]["first"])
                        <= min(e["last"], observed_events[i]["last"])
                        for i in actual_ids
                    )
                )
            )
        )
        if old == (len(actual_ids), len(predicted_ids)):
            break
    reference_incomplete = any((not observed_events[i]["complete"] for i in actual_ids))
    prediction_incomplete = any(
        (not predicted_events[i]["complete"] for i in predicted_ids)
    )
    actual = [
        e for i, e in enumerate(observed_events) if i in actual_ids and e["complete"]
    ]
    predicted = [
        e
        for j, e in enumerate(predicted_events)
        if j in predicted_ids and e["complete"]
    ]
    possible = sorted(
        (
            (abs(t["peak"] - p["peak"]), i, j)
            for i, t in enumerate(actual)
            for j, p in enumerate(predicted)
            if max(t["first"], p["first"]) <= min(t["last"], p["last"])
        )
    )
    used_t, used_p, matches = (set(), set(), [])
    for timing, i, j in possible:
        if i in used_t or j in used_p:
            continue
        used_t.add(i)
        used_p.add(j)
        t, p = (actual[i], predicted[j])
        matches.append(
            {
                "key": origin + t["peak"],
                "predicted_key": origin + p["peak"],
                "peak_error": abs(estimate[p["peak"]] - truth[t["peak"]]),
                "timing": timing,
                "ratio": truth[t["peak"]] / estimate[p["peak"]],
                "estimated_peak": bool(np.any(changed == p["peak"])),
            }
        )
    return {
        "matches": matches,
        "missed": [origin + t["peak"] for i, t in enumerate(actual) if i not in used_t],
        "false": [
            origin + p["peak"] for j, p in enumerate(predicted) if j not in used_p
        ],
        "reference_incomplete": reference_incomplete,
        "prediction_incomplete": prediction_incomplete,
    }


def exposure_years(values):
    return float(np.isfinite(values).sum() / (24 * 365.2425))


def create_regression(index, frame):
    model = {}
    model["index"] = index
    model["y"] = frame[target_station].to_numpy(float)
    model["years"] = np.unique(index.year)
    model["year"] = index.year.to_numpy()
    model["group"] = np.searchsorted(model["years"], model["year"])
    model["values"] = {sid: frame[sid].to_numpy(float) for sid in donor_stations}
    model["stats"] = {}
    model["shift_cache"] = {}
    model["fit_cache"] = {}
    return model


def shift_donor(model, sid, lag):
    key = (sid, lag)
    if key in model["shift_cache"]:
        return model["shift_cache"][key]
    result = np.full(len(model["y"]), np.nan)
    if lag == 0:
        return model["values"][sid]
    if lag > 0:
        result[lag:] = model["values"][sid][:-lag]
    else:
        result[:lag] = model["values"][sid][-lag:]
    return model["shift_cache"].setdefault(key, result)


def regression_design(model, spec, start=0, stop=None):
    stop = len(model["y"]) if stop is None else stop
    return np.column_stack(
        [np.ones(stop - start)]
        + [
            shift_donor(model, sid, lag)[start:stop]
            for sid, lag in zip(spec[0], spec[1])
        ]
    )


def regression_sums(model, spec):
    if spec in model["stats"]:
        return model["stats"][spec]
    X = regression_design(model, spec)
    valid = np.isfinite(model["y"]) & np.isfinite(X).all(axis=1)
    positions = np.arange(len(model["y"]))
    for lag in spec[1]:
        source = np.clip(positions - lag, 0, len(model["y"]) - 1)
        valid &= model["year"][source] == model["year"]
    x, y, g = (X[valid], model["y"][valid], model["group"][valid])
    n, p = (len(model["years"]), X.shape[1])
    xx, xy = (np.zeros((n, p, p)), np.zeros((n, p)))
    for i in range(p):
        xy[:, i] = np.bincount(g, weights=x[:, i] * y, minlength=n)
        for j in range(p):
            xx[:, i, j] = np.bincount(g, weights=x[:, i] * x[:, j], minlength=n)
    result = (
        xx,
        xy,
        np.bincount(g, weights=y * y, minlength=n),
        np.bincount(g, minlength=n),
    )
    model["stats"][spec] = result
    return result


def regression_score(model, spec, held):
    key = (spec, tuple(held))
    cache = model.setdefault("score_cache", {})
    if key in cache:
        return cache[key]
    xx, xy, yy, count = regression_sums(model, spec)
    allowed = ~np.isin(model["years"], held)
    total_x, total_y = (xx[allowed].sum(axis=0), xy[allowed].sum(axis=0))
    errors = []
    for i in np.flatnonzero(allowed & (count >= 240)):
        train = allowed.copy()
        train[i] = False
        if count[train].sum() < 720 or (count[train] > 0).sum() < 3:
            continue
        beta = solve_regression(total_x - xx[i], total_y - xy[i])
        if beta is not None:
            loss = yy[i] - 2 * beta @ xy[i] + beta @ xx[i] @ beta
            errors.append(np.sqrt(max(loss, 0) / count[i]))
    cache[key] = float(np.mean(errors)) if len(errors) >= 4 else np.inf
    return cache[key]


def solve_regression(mat, rhs):
    scale = np.sqrt(np.maximum(np.diag(mat), 1e-12))
    norm = mat / np.outer(scale, scale)
    if np.linalg.cond(norm) > 100000000.0:
        return None
    return np.linalg.solve(norm, rhs / scale) / scale


def fit_regression(model, spec, held):
    key = (spec, tuple(held))
    cache = model["fit_cache"]
    if key in cache:
        return cache[key]
    xx, xy, _, count = regression_sums(model, spec)
    use = ~np.isin(model["years"], held)
    if count[use].sum() < 720 or (count[use] > 0).sum() < 3:
        return None
    cache[key] = solve_regression(xx[use].sum(axis=0), xy[use].sum(axis=0))
    return cache[key]


def predict_regression(model, spec, held, start, stop):
    beta = fit_regression(model, spec, tuple(sorted(held)))
    return (
        np.full(stop - start, np.nan)
        if beta is None
        else regression_design(model, spec, start, stop) @ beta
    )


def create_reconstruction():
    data = {}
    source = pd.read_csv(
        raw_folder / "2.11.0-Vannføring-time-v0.csv", sep=";", decimal=",", skiprows=1
    )
    source.index = pd.to_datetime(source.Tidspunkt, utc=True)
    data["index"] = pd.date_range(
        "1987-01-01", "2026-01-01", freq="h", inclusive="left", tz="UTC"
    )
    data["hourly"] = source.reindex(data["index"])
    data["n"] = len(data["index"])
    data["year"] = data["index"].year.to_numpy()
    data["years"] = set(data["year"])
    data["allowed"] = ~np.isin(data["year"], list(excluded_years))
    raw = data["hourly"]["Vannføring (m³/s)"].to_numpy(float)
    accepted = (
        data["hourly"].Kontrollert.eq(3) & data["hourly"].Korrigert.isin([0, 1])
    ).to_numpy()
    quality_observed = np.where(accepted & (raw >= 0), raw, np.nan)
    data["q"] = np.where(data["allowed"], quality_observed, np.nan)
    data["raw"] = raw
    before = data["q"][np.isin(data["year"], reference_years)]
    data["threshold"] = float(np.nanquantile(before, 0.98))
    data["right_anchors"] = {}
    data["boundary_anchors"] = {}
    for stamp in boundary_dates:
        i = data["index"].get_loc(pd.Timestamp(stamp))
        if accepted[i] and np.isfinite(raw[i]) and (raw[i] >= 0):
            data["right_anchors"][i] = raw[i]
            data["boundary_anchors"][i] = raw[i]
    data["daily"], data["daily_flags"] = ({}, {})
    for start, value, correction, control in daily_source():
        i = int((start - data["index"][0]).total_seconds() // 3600)
        if control == "3" and correction in {"0", "1"}:
            data["daily"][i] = (value, correction)
            data["daily_flags"][i] = correction
    data["components"], rows = ([], [])
    source_runs = runs(~np.isfinite(quality_observed))
    source_starts = np.array([a for a, b in source_runs])
    for gap_id, (a, b) in enumerate(runs(data["allowed"] & ~np.isfinite(data["q"])), 1):
        pa, pb = source_runs[np.searchsorted(source_starts, a, side="right") - 1]
        assert pa <= a < b <= pb
        if pb - pa <= maximum_gap:
            pattern_a, pattern_b = (pa, pb)
            for i in (pa - 1, pb):
                if (
                    0 <= i < data["n"]
                    and (not data["allowed"][i])
                    and np.isfinite(quality_observed[i])
                ):
                    data["boundary_anchors"][i] = quality_observed[i]
                    data["right_anchors"][i] = quality_observed[i]
        else:
            pattern_a, pattern_b = (a, b)
        rows.append(
            dict(
                gap_id=gap_id,
                a=a,
                b=b,
                start_utc=data["index"][a],
                end_utc=data["index"][b - 1],
                missing_hours=b - a,
                source_pattern_start_utc=data["index"][pa],
                source_pattern_end_utc=data["index"][pb - 1],
                source_pattern_hours=pb - pa,
            )
        )
        days = daily_starts(pattern_a, pattern_b - 1)
        if data["components"] and days[0] <= data["components"][-1][0][-1]:
            old_days, blocks = data["components"][-1]
            data["components"][-1] = (
                np.union1d(old_days, days),
                blocks + [(gap_id, pattern_a, pattern_b)],
            )
        else:
            data["components"].append((days, [(gap_id, pattern_a, pattern_b)]))
    data["gaps"] = pd.DataFrame(rows)
    inputs = read_inputs()
    donors = inputs["corrected"].reindex(data["index"])[list(donor_stations)].copy()
    unset = inputs["aligned"]["2.265.0"].reindex(data["index"])
    donors["2.265.0"] = donors["2.265.0"].where(
        unset.Kontrollert.eq(3) & ~np.isin(data["year"], list(excluded_donor_years))
    )
    donors.insert(0, target_station, data["q"])
    data["models"] = create_regression(data["index"], donors)
    data["reference_events"] = [
        e for e in extract(data["q"], data["threshold"]) if e["complete"]
    ]
    finite_runs = runs(np.isfinite(data["q"]))
    data["finite_starts"] = np.array([a for a, b in finite_runs])
    data["finite_stops"] = np.array([b for a, b in finite_runs])
    data["missing_prefix"] = np.r_[0, np.cumsum(~np.isfinite(data["q"]))]
    data["validation"], data["window_results"], data["training_audit"] = ([], [], [])
    assert (
        data["gaps"].missing_hours.sum()
        == (data["allowed"] & ~np.isfinite(data["q"])).sum()
    )
    return data


def daily_starts(first, last):
    starts = np.arange((first + 1) // 24 * 24 - 1, last + 1, 24)
    if (first + 1) % 24 == 0:
        starts = np.r_[starts[0] - 24, starts]
    return starts


def daily_adjust(base, positions, constraints, edge_correction):
    """Smooth the correction subject to the weighted reporting-day budgets."""
    n = len(positions)
    row, col, weights = ([], [], [])
    budgets = []
    for j, (_, _, indices, w, budget) in enumerate(constraints):
        row.extend([j] * len(indices))
        col.extend(indices)
        weights.extend(w)
        budgets.append(budget)
    matrix = sparse.csc_matrix((weights, (row, col)), shape=(len(constraints), n))
    links = (np.diff(positions) == 1).astype(float)
    smooth = sparse.diags([-links, np.full(n, 2.0), -links], [-1, 0, 1], format="csc")
    rhs = smooth @ base + edge_correction
    budgets = np.asarray(budgets)
    system = sparse.bmat([[smooth, matrix.T], [matrix, None]], format="csc")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MatrixRankWarning)
        estimate = spsolve(system, np.r_[rhs, budgets])[:n]
    if np.isfinite(estimate).all() and estimate.min() < -1e-08:
        from scipy.optimize import Bounds, LinearConstraint, minimize

        if n > 160:
            return None
        constraint = LinearConstraint(matrix.toarray(), budgets, budgets)
        objective = lambda y: 0.5 * y @ (smooth @ y) - rhs @ y
        derivative = lambda y: smooth @ y - rhs
        fit = minimize(
            objective,
            np.maximum(estimate, 0),
            jac=derivative,
            bounds=Bounds(0, np.inf),
            constraints=constraint,
            method="SLSQP",
            options={"maxiter": 80, "ftol": 1e-09},
        )
        estimate = fit.x if fit.success else np.full(n, np.nan)
    if not np.isfinite(estimate).all() or estimate.min() < -1e-08:
        return None
    estimate = np.maximum(estimate, 0)
    if not np.allclose(matrix @ estimate, budgets, rtol=1e-08, atol=1e-07):
        return None
    return estimate


def make_prediction(values, donor_spec="", constraints=(), excluded_right_anchors=()):
    return {
        "values": values,
        "donor_spec": donor_spec,
        "constraints": constraints,
        "excluded_right_anchors": excluded_right_anchors,
    }


def donor_ranking(data, held, donor_sizes=(1, 2)):
    selected = {}
    for donor in donor_stations:
        scores = []
        for lag in lag_candidates:
            spec = ((donor,), (lag,))
            scores.append((blocked_score(data, spec, held), abs(lag), lag))
        score, _, lag = min(scores)
        if np.isfinite(score):
            selected[donor] = lag
    ranked = []
    for size in donor_sizes:
        for stations in combinations(selected, size):
            spec = (stations, tuple((selected[sid] for sid in stations)))
            score = blocked_score(data, spec, held)
            if np.isfinite(score):
                ranked.append((score, size, spec))
    return [spec for _, _, spec in sorted(ranked, key=lambda row: row[:2])]


def donor_prediction(data, positions, held, boundary_hours=None, donor_sizes=(1, 2)):
    a, b = (positions[0], positions[-1] + 1)
    start, stop = (max(0, a - 1), min(data["n"], b + 1))
    for spec in donor_ranking(data, held, donor_sizes):
        q = predict_regression(data["models"], spec, held, start, stop)
        values = q[positions - start]
        if not np.isfinite(values).all() or np.min(values) < 0:
            continue
        if boundary_hours is not None and (
            not np.isfinite(q[boundary_hours - start]).all()
        ):
            continue
        label = ", ".join(
            (
                f"{donor_stations[sid]} ({lag:+d} h)"
                for sid, lag in zip(spec[0], spec[1])
            )
        )
        return (values, q, start, label)
    return None


def daily_constraints(data, positions):
    constraints = []
    for day in daily_starts(positions[0], positions[-1]):
        if day < 0 or day + 24 >= data["n"] or day not in data["daily"]:
            continue
        mean, code = data["daily"][day]
        sample = np.arange(day, day + 25)
        w = np.r_[0.5, np.ones(23), 0.5]
        hidden = np.isin(sample, positions)
        if not hidden.any() or not np.isfinite(data["q"][sample[~hidden]]).all():
            continue
        if w[hidden].sum() == 0.5:
            continue
        budget = 24 * mean - w[~hidden] @ data["q"][sample[~hidden]]
        if budget < -1e-08:
            return None
        constraints.append(
            (
                day,
                code,
                np.searchsorted(positions, sample[hidden]),
                w[hidden],
                max(0.0, budget),
            )
        )
    return tuple(constraints)


def initial_predictions(
    data, positions, blocks, held, methods=methods, donor_sizes=(1, 2)
):
    a, b = (positions[0], positions[-1] + 1)
    predictions = {}
    linear = []
    for first, last in blocks:
        if (
            first == 0
            or last == data["n"]
            or (not np.isfinite(data["q"][[first - 1, last]]).all())
        ):
            linear = None
            break
        linear.extend(
            np.linspace(data["q"][first - 1], data["q"][last], last - first + 2)[1:-1]
        )
    if linear is not None and "linear" in methods:
        predictions["linear"] = make_prediction(np.asarray(linear))
    if "donor_ols" in methods:
        donor = donor_prediction(data, positions, held, donor_sizes=donor_sizes)
        if donor is not None:
            predictions["donor_ols"] = make_prediction(donor[0], donor[3])
    if not set(methods) & {"daily_linear", "daily_donor_ols"}:
        return predictions
    constraints = daily_constraints(data, positions)
    if not constraints:
        return predictions
    if "daily_linear" in methods and linear is not None:
        estimate = daily_adjust(
            np.asarray(linear), positions, constraints, np.zeros(len(positions))
        )
        if estimate is not None:
            predictions["daily_linear"] = make_prediction(
                estimate, constraints=constraints
            )
    if "daily_donor_ols" in methods and a > 0 and (b < data["n"]):
        first, last = np.asarray(blocks).T
        boundary_hours = np.r_[first - 1, last]
        right_observed = [
            data["right_anchors"].get(hour, data["q"][hour]) for hour in last
        ]
        observed = np.r_[data["q"][first - 1], right_observed]
        if not np.isfinite(observed).all():
            return predictions
        donor = donor_prediction(data, positions, held, boundary_hours, donor_sizes)
        if donor is None:
            return predictions
        base, donor_window, start, label = donor
        corrections = observed - donor_window[boundary_hours - start]
        edge = np.zeros(len(positions))
        np.add.at(edge, np.searchsorted(positions, np.r_[first, last - 1]), corrections)
        estimate = daily_adjust(base, positions, constraints, edge)
        if estimate is not None:
            excluded_anchors = tuple(
                (hour for hour in last if hour in data["right_anchors"])
            )
            predictions["daily_donor_ols"] = make_prediction(
                estimate, label, constraints, excluded_anchors
            )
    return predictions


def blocked_score(data, spec, held):
    return regression_score(data["models"], spec, held)


def boundary_context(data, blocks):
    needed = {i for a, b in blocks for i in (a - 1, b) if i in data["boundary_anchors"]}
    if not needed:
        return data["q"]
    q = data["q"].copy()
    for i in needed:
        q[i] = data["boundary_anchors"][i]
    return q


def predict(data, positions, blocks, held, allowed_daily_days=None, **kwargs):
    observed, daily = (data["q"], data["daily"])
    try:
        data["q"] = boundary_context(data, blocks)
        if allowed_daily_days is not None:
            data["daily"] = {d: daily[d] for d in allowed_daily_days if d in daily}
        return initial_predictions(data, positions, blocks, held, **kwargs)
    finally:
        data["q"], data["daily"] = (observed, daily)


def reference_context(data, first, stop):
    key = (first, stop)
    cache = data.setdefault("context_cache", {})
    if key in cache:
        return cache[key]
    if context_bounds(data["q"], first, stop, data["threshold"]) is None:
        cache[key] = None
        return None
    i = np.searchsorted(data["finite_starts"], first, side="right") - 1
    cache[key] = (int(data["finite_starts"][i]), int(data["finite_stops"][i]))
    return cache[key]


def target_context(data, positions, predictions):
    bounds = []
    for p in predictions.values():
        b = prediction_context(data, positions, p["values"])
        if b is not None:
            bounds.append(b)
    return (
        (min((b[0] for b in bounds)), max((b[1] for b in bounds))) if bounds else None
    )


def prediction_context(data, positions, values):
    q = data["q"].copy()
    q[positions] = values
    relevant = [
        e
        for e in extract(q, data["threshold"])
        if np.any(
            (positions >= e["first"] - separation_hours)
            & (positions <= e["last"] + separation_hours)
        )
    ]
    if not relevant:
        return context_bounds(
            q, int(positions[0]), int(positions[-1] + 1), data["threshold"]
        )
    if not all((e["complete"] for e in relevant)):
        return None
    bounds = [
        context_bounds(q, e["first"], e["last"] + 1, data["threshold"])
        for e in relevant
    ]
    if any((b is None for b in bounds)):
        return None
    left, right = (min((b[0] for b in bounds)), max((b[1] for b in bounds)))
    return (left, right)


def regime(data, a, b):
    values = [data["daily"][d][0] for d in daily_starts(a, b - 1) if d in data["daily"]]
    return flow_regime(values)


def flow_regime(values):
    if not values:
        return (None, "unknown")
    if len(values) < 2:
        return (max(values), "unknown")
    movement = (values[-1] - values[0]) / max(np.mean(values), 0.05)
    return (
        max(values),
        "rising" if movement > 0.2 else "falling" if movement < -0.2 else "steady",
    )


def matching_daily_offsets(data, first, stop):
    return tuple(
        (int(d - first) for d in daily_starts(first, stop - 1) if d in data["daily"])
    )


def comparable_daily_days(data, test_first, real_first, real_stop):
    return tuple(
        (
            test_first + offset
            for offset in matching_daily_offsets(data, real_first, real_stop)
        )
    )


def comparable_regime(data, test_first, real_first, real_stop):
    days = comparable_daily_days(data, test_first, real_first, real_stop)
    if any((d not in data["daily"] for d in days)):
        return None
    return flow_regime([data["daily"][d][0] for d in days])


def observed_boundaries(data, positions):
    starts = positions[np.r_[True, np.diff(positions) > 1]] - 1
    stops = positions[np.r_[np.diff(positions) > 1, True]] + 1
    values = []
    for i in np.unique(np.r_[starts, stops]):
        if not 0 <= i < data["n"]:
            continue
        q = data["boundary_anchors"].get(int(i), data["q"][i])
        if np.isfinite(q):
            values.append((int(i - positions[0]), float(q)))
    return values


def boundary_distance(data, test_first, profile):
    differences = [
        abs(np.log((data["q"][test_first + offset] + 0.05) / (real + 0.05)))
        for offset, real in profile
    ]
    return float(np.mean(differences)) if differences else 0.0


def find_test_periods(data, positions, held, radius, need_events):
    first, stop = (int(positions[0]), int(positions[-1] + 1))
    width = stop - first
    target = data["index"][first]
    level, direction = regime(data, first, stop)
    boundary_profile = observed_boundaries(data, positions)
    candidates = {}
    offsets = [0] + [d for i in range(1, radius + 1) for d in (-i, i)]
    for year in sorted(data["years"] - set(held)):
        anchor = target.replace(
            year=int(year),
            day=28 if target.month == 2 and target.day == 29 else target.day,
        )
        center = data["index"].get_indexer([anchor])[0]
        for days in offsets:
            c, end = (center + 24 * days, center + 24 * days + width)
            if c < 25 or end + 25 >= data["n"]:
                continue
            if data["missing_prefix"][end + 1] != data["missing_prefix"][c - 1]:
                continue
            matched = comparable_regime(data, c, first, stop)
            if matched is None:
                continue
            match, movement = matched
            if level is not None and (
                match is None
                or (direction != "unknown" and movement != direction)
                or (not 0.5 <= (match + 0.05) / (level + 0.05) <= 2)
            ):
                continue
            distance = (
                0.0 if level is None else abs(np.log((match + 0.05) / (level + 0.05)))
            )
            distance += boundary_distance(data, c, boundary_profile)
            candidates[c] = (False, abs(days), distance)
    if need_events:
        for event in data["reference_events"]:
            c = event["peak"] - width // 2
            end = c + width
            if c < 25 or end + 25 >= data["n"]:
                continue
            if data["missing_prefix"][end + 1] != data["missing_prefix"][c - 1]:
                continue
            date = target.replace(
                year=data["index"][c].year,
                day=28 if target.month == 2 and target.day == 29 else target.day,
            )
            days = abs((data["index"][c] - date).total_seconds() / 86400)
            if days > radius:
                continue
            matched = comparable_regime(data, c, first, stop)
            if matched is None:
                continue
            match, _ = matched
            if level is not None and (
                match is None or not 0.5 <= (match + 0.05) / (level + 0.05) <= 2
            ):
                continue
            distance = (
                0.0 if level is None else abs(np.log((match + 0.05) / (level + 0.05)))
            )
            distance += boundary_distance(data, c, boundary_profile)
            candidates[c] = (True, days, distance)
    selected, used, per_year = ([], [], Counter())
    ordered = sorted(
        candidates,
        key=lambda c: (not candidates[c][0], candidates[c][1], candidates[c][2], c),
    )
    for c in ordered:
        bounds = reference_context(data, c, c + width)
        if bounds is None:
            bounds = (c - 25, c + width + 25)
        left, right = bounds
        if set(data["year"][c : c + width]) & set(held):
            continue
        if any((max(c, l) < min(c + width, r) for l, r in used)):
            continue
        stress, days, distance = candidates[c]
        if not stress and per_year[data["year"][c]] >= 3:
            continue
        selected.append((c, stress, days, distance, bounds))
        used.append((c, c + width))
        if not stress:
            per_year[data["year"][c]] += 1
    return selected


def test_case(data, c, positions, blocks, held, actual, bounds, rank):
    shift = c - int(positions[0])
    hidden = positions + shift
    left, right = bounds
    fold = tuple(sorted(set(held) | set(data["year"][left:right])))
    assert not set(held) & set(data["year"][hidden])
    days = comparable_daily_days(data, c, int(positions[0]), int(positions[-1] + 1))
    predicted = predict(
        data,
        hidden,
        [(a + shift, b + shift) for a, b in blocks],
        fold,
        methods=tuple(actual),
        allowed_daily_days=days,
    )
    data["training_audit"].append(
        dict(
            actual_component=int(positions[0]),
            test_start=c,
            excluded_years=",".join(map(str, fold)),
            context_start=left,
            context_stop=right,
            context_years_excluded=True,
            lag_year_crossings_removed=True,
            matching_daily_offsets_h=",".join((str(d - c) for d in days)),
            daily_input_mask_applied=True,
        )
    )
    y = data["q"][hidden]
    records = {}
    for method, p in predicted.items():
        if p["constraints"]:
            fraction = lambda x: len(
                {int(i) for z in x["constraints"] for i in z[2]}
            ) / len(hidden)
            if abs(fraction(p) - fraction(actual[method])) > 0.2:
                continue
        fit = p["values"]
        event_result, event_available, event_unresolved = score_event_case(
            data, hidden, fit, fold
        )
        record = dict(
            start=c,
            year=int(data["year"][c]),
            rank=rank,
            mae=float(np.abs(fit - y).mean() / max(y.mean(), 1e-08)),
            mae_m3_s=float(np.abs(fit - y).mean()),
            peak_error=float(abs(fit.max() - y.max()) / max(y.max(), 1e-08)),
            peak_bias=float((fit.max() - y.max()) / max(y.max(), 1e-08)),
            peak_residual=float(y.max() - fit.max()),
            timing=abs(int(fit.argmax()) - int(y.argmax())),
            events=event_result,
            matching_daily_values=len(days),
            daily_input_mask_applied=True,
            event_context_available=event_available,
            event_context_unresolved=event_unresolved,
        )
        records[method] = record
    return records


def score_event_case(data, hidden, fit, fold):
    first, stop = (int(hidden[0]), int(hidden[-1] + 1))
    bounds = reference_context(data, first, stop)
    empty = dict(matches=[], missed=[], false=[])
    if bounds is None:
        return (empty, False, False)
    left, right = bounds
    assert set(data["year"][left:right]).issubset(fold)
    reference = data["q"][left:right]
    estimate = reference.copy()
    estimate[hidden - left] = fit
    result = compare(reference, estimate, hidden - left, data["threshold"], left)
    if result["reference_incomplete"]:
        return (empty, False, False)
    return (result, True, result["prediction_incomplete"])


def score(rows):
    result = dict(
        test_windows=len(rows),
        test_years=len({x["year"] for x in rows}),
        event_test_windows=sum((r.get("event_context_available", True) for r in rows)),
        unresolved_event_tests=sum(
            (r.get("event_context_unresolved", False) for r in rows)
        ),
        window_ids={x["start"] for x in rows},
        window_mae={x["start"]: x["mae"] for x in rows},
    )
    for key, field, p in [
        ("relative_mae", "mae", 0.5),
        ("median_peak_error", "peak_error", 0.5),
        ("p90_peak_error", "peak_error", 0.9),
        ("upper_peak_residual", "peak_residual", 0.95),
        ("maximum_peak_residual", "peak_residual", 1.0),
        ("median_peak_bias", "peak_bias", 0.5),
        ("median_timing_h", "timing", 0.5),
    ]:
        result[key] = (
            float(np.quantile([r[field] for r in rows], p)) if rows else np.nan
        )
    truth, false, representatives = ({}, {}, set())
    for row in sorted(rows, key=lambda r: r["rank"]):
        e = row["events"]
        for match in e["matches"]:
            if match["key"] not in truth:
                representatives.add(row["start"])
            truth.setdefault(match["key"], match)
        for key in e["missed"]:
            if key not in truth:
                representatives.add(row["start"])
            truth.setdefault(key, None)
    for row in sorted(rows, key=lambda r: r["rank"]):
        e = row["events"]
        if (e["matches"] or e["missed"]) and row["start"] not in representatives:
            continue
        for key in e["false"]:
            false.setdefault(key, row["start"])
    hits = [x for x in truth.values() if x is not None]
    ratios = [x["ratio"] for x in hits if x["estimated_peak"]]
    n, h, f = (len(truth), len(hits), len(false))
    result.update(
        true_events=n,
        hits=h,
        misses=n - h,
        false_events=f,
        recall=h / n if n else np.nan,
        precision=h / (h + f) if h + f else np.nan,
        matched_estimated_peaks=len(ratios),
        factor_p05=float(np.quantile(ratios, 0.05)) if ratios else np.nan,
        factor_p95=float(np.quantile(ratios, 0.95)) if ratios else np.nan,
    )
    return result


def assess(data, group, methods=methods, fixed_methods=None):
    blocks = [(a, b) for _, a, b in group]
    positions = np.concatenate([np.arange(a, b) for a, b in blocks])
    first, stop = (int(positions[0]), int(positions[-1] + 1))
    held = tuple(
        sorted(
            set(
                data["year"][
                    max(0, first - separation_hours) : min(
                        data["n"], stop + separation_hours
                    )
                ]
            )
        )
    )
    actual = predict(data, positions, blocks, held, methods=methods)
    if fixed_methods is not None:
        actual = combined_prediction(data, group, held, fixed_methods)
    context = target_context(data, positions, actual)
    if context is not None:
        expanded = tuple(sorted(set(held) | set(data["year"][context[0] : context[1]])))
        if expanded != held:
            held = expanded
            actual = (
                predict(data, positions, blocks, held, methods=methods)
                if fixed_methods is None
                else combined_prediction(data, group, held, fixed_methods)
            )
            context = target_context(data, positions, actual)
    if not actual:
        return (positions, None, None, "no_feasible_prediction")
    near = data["q"][
        max(0, first - separation_hours) : min(data["n"], stop + separation_hours)
    ]
    level, _ = regime(data, first, stop)
    need_events = bool(
        np.any(near > data["threshold"])
        or (level is not None and level >= 12.0)
        or any((p["values"].max() > data["threshold"] for p in actual.values()))
    )
    cached = {}
    for radius in (28, 56):
        windows = find_test_periods(data, positions, held, radius, need_events)
        records = defaultdict(list)
        for c, stress, days, distance, bounds in windows:
            if c not in cached:
                rank = (days, distance, c)
                cached[c] = (
                    test_case(data, c, positions, blocks, held, actual, bounds, rank)
                    if fixed_methods is None
                    else test_combined(
                        data,
                        c,
                        positions,
                        group,
                        held,
                        fixed_methods,
                        bounds,
                        rank,
                        actual["combined"],
                    )
                )
            for method, row in cached[c].items():
                records[method].append(row)
        scores = {m: score(records[m]) for m in actual}
        if all(
            (
                s["test_windows"] >= minimum_tests
                and s["test_years"] >= minimum_years
                and (not need_events or s["true_events"] >= minimum_events)
                for s in scores.values()
            )
        ):
            break
    candidates, pot = ([], [])
    for method, p in actual.items():
        s = scores[method]
        enough = s["test_windows"] >= minimum_tests and s["test_years"] >= minimum_years
        accurate = enough and all(
            (
                s[k] <= v
                for k, v in zip(
                    ("relative_mae", "median_peak_error", "p90_peak_error"),
                    error_limits,
                )
            )
        )
        upper = (
            float(p["values"].max() + max(0.0, s["upper_peak_residual"]))
            if enough
            else np.inf
        )
        risk, risk_basis = decision_sensitive(
            boundary_context(data, blocks), positions, blocks, upper, data["threshold"]
        )
        support = (
            s["true_events"] >= minimum_events
            and s["recall"] >= 0.8
            and (s["precision"] >= 0.8)
            and (s["unresolved_event_tests"] == 0)
        )
        interval_ok = True
        copy = data["q"].copy()
        copy[positions] = p["values"]
        actual_context = prediction_context(data, positions, p["values"])
        if actual_context is not None:
            for e in affected(extract(copy, data["threshold"]), positions):
                if np.any(positions == e["peak"]):
                    interval_ok &= (
                        s["matched_estimated_peaks"] >= minimum_events
                        and copy[e["peak"]] * s["factor_p05"] > data["threshold"]
                    )
        approved = accurate and (
            not risk or (support and interval_ok and (actual_context is not None))
        )
        reason = (
            "insufficient_comparable_windows"
            if not enough
            else (
                "hourly_or_peak_error_above_limit"
                if not accurate
                else (
                    "low_risk_on_empirical_screen"
                    if not risk
                    else (
                        "insufficient_event_support_or_skill"
                        if not support
                        else (
                            "estimated_peak_interval_crosses_threshold"
                            if not interval_ok
                            else (
                                "incomplete_actual_event_context"
                                if actual_context is None
                                else "event_validation_passed"
                            )
                        )
                    )
                )
            )
        )
        row = dict(
            component_id=group[0][0],
            scope="combined" if fixed_methods else "component",
            method=method,
            start_utc=data["index"][first],
            hidden_hours=len(positions),
            search_radius_days=radius,
            held_years=",".join(map(str, held)),
            **{k: v for k, v in s.items() if k not in {"window_ids", "window_mae"}},
            upper_peak_screen_m3_s=upper,
            flood_sensitive=risk,
            predicted_peak_m3_s=float(p["values"].max()),
            risk_screen_basis=risk_basis,
            hourly_pass=accurate,
            pot_pass=approved,
            reason=reason,
            selected=False,
            donors_and_lags=p["donor_spec"],
        )
        row["matching_daily_offsets_h"] = ",".join(
            map(str, matching_daily_offsets(data, first, stop))
        )
        row["daily_information_matching"] = "same_reporting_days_as_real_gap"
        data["validation"].append(row)
        for r in records[method]:
            data["window_results"].append(
                dict(
                    component_id=group[0][0],
                    scope=row["scope"],
                    method=method,
                    test_start_utc=data["index"][r["start"]],
                    test_year=r["year"],
                    **{
                        k: r[k]
                        for k in [
                            "mae",
                            "mae_m3_s",
                            "peak_error",
                            "peak_bias",
                            "timing",
                        ]
                    },
                    true_events=len(r["events"]["matches"])
                    + len(r["events"]["missed"]),
                    event_context_available=r.get("event_context_available", True),
                    event_context_unresolved=r.get("event_context_unresolved", False),
                    matching_daily_values=r["matching_daily_values"],
                    daily_input_mask_applied=r["daily_input_mask_applied"],
                    hits=len(r["events"]["matches"]),
                    false_events=len(r["events"]["false"]),
                )
            )
        option = (method, p, s, row)
        if accurate:
            candidates.append(option)
        if approved:
            pot.append(option)

    def choose(options):
        if not options:
            return None
        common = set.intersection(*(x[2]["window_ids"] for x in options))
        paired = (
            len(common) >= minimum_tests
            and len({data["year"][c] for c in common}) >= minimum_years
        )
        losses = {
            x[0]: (
                float(np.median([x[2]["window_mae"][c] for c in common]))
                if paired
                else x[2]["relative_mae"]
            )
            for x in options
        }
        best = min(losses.values())
        near = [x for x in options if losses[x[0]] <= 1.05 * best]
        chosen = min(
            near,
            key=lambda x: (methods.index(x[0]) if x[0] in methods else 4, losses[x[0]]),
        )
        chosen[3]["selected"] = True
        chosen[3]["selection_shared_windows"] = len(common) if paired else 0
        return chosen

    return (positions, choose(candidates), choose(pot), "assessed")


def combined_prediction(data, group, held, fixed_methods, allowed_daily_days=None):
    ordered = sorted(group, key=lambda g: g[1])
    positions = np.concatenate([np.arange(a, b) for _, a, b in ordered])
    values = np.full(len(positions), np.nan)
    coupled = []
    for item in ordered:
        days = daily_starts(item[1], item[2] - 1)
        if coupled and days[0] <= coupled[-1][0][-1]:
            previous, items = coupled[-1]
            coupled[-1] = (np.union1d(previous, days), items + [item])
        else:
            coupled.append((days, [item]))
    labels, constraints = ([], [])
    for _, items in coupled:
        method_set = {fixed_methods[gap_id] for gap_id, _, _ in items}
        if len(method_set) != 1:
            return {}
        method = method_set.pop()
        subpositions = np.concatenate([np.arange(a, b) for _, a, b in items])
        blocks = [(a, b) for _, a, b in items]
        p = predict(
            data,
            subpositions,
            blocks,
            held,
            methods=(method,),
            allowed_daily_days=allowed_daily_days,
        ).get(method)
        if p is None:
            return {}
        values[np.searchsorted(positions, subpositions)] = p["values"]
        labels.append(p["donor_spec"])
        for day, code, indices, w, budget in p["constraints"]:
            constraints.append(
                (
                    day,
                    code,
                    np.searchsorted(positions, subpositions[indices]),
                    w,
                    budget,
                )
            )
    return {"combined": make_prediction(values, "; ".join(labels), tuple(constraints))}


def test_combined(
    data, c, positions, group, held, fixed_methods, bounds, rank, reference_prediction
):
    shift = c - int(positions[0])
    left, right = bounds
    fold = tuple(sorted(set(held) | set(data["year"][left:right])))
    shifted = [(gap_id, a + shift, b + shift) for gap_id, a, b in group]
    days = comparable_daily_days(data, c, int(positions[0]), int(positions[-1] + 1))
    actual = combined_prediction(
        data, shifted, fold, fixed_methods, allowed_daily_days=days
    )
    if not actual:
        return {}
    fraction = lambda p: len(
        {int(i) for item in p["constraints"] for i in item[2]}
    ) / len(positions)
    if abs(fraction(actual["combined"]) - fraction(reference_prediction)) > 0.2:
        return {}
    data["training_audit"].append(
        dict(
            actual_component=int(positions[0]),
            test_start=c,
            excluded_years=",".join(map(str, fold)),
            context_start=left,
            context_stop=right,
            context_years_excluded=True,
            lag_year_crossings_removed=True,
            scope="combined",
            matching_daily_offsets_h=",".join((str(d - c) for d in days)),
            daily_input_mask_applied=True,
        )
    )
    hidden = positions + shift
    y = data["q"][hidden]
    fit = actual["combined"]["values"]
    result, available, unresolved = score_event_case(data, hidden, fit, fold)
    return {
        "combined": dict(
            start=c,
            year=int(data["year"][c]),
            rank=rank,
            mae=float(np.abs(fit - y).mean() / max(y.mean(), 1e-08)),
            mae_m3_s=float(np.abs(fit - y).mean()),
            peak_error=float(abs(fit.max() - y.max()) / max(y.max(), 1e-08)),
            peak_bias=float((fit.max() - y.max()) / max(y.max(), 1e-08)),
            peak_residual=float(y.max() - fit.max()),
            timing=abs(int(fit.argmax()) - int(y.argmax())),
            events=result,
            matching_daily_values=len(days),
            daily_input_mask_applied=True,
            event_context_available=available,
            event_context_unresolved=unresolved,
        )
    }


reconstruction_folder = output_folder / "reconstruction"


def verify_prediction(data, positions, option):
    if option is None:
        return
    _, prediction, _, _ = option
    assert np.isfinite(prediction["values"]).all() and np.all(prediction["values"] >= 0)
    for _, _, indices, weights, budget in prediction["constraints"]:
        assert np.isclose(
            weights @ prediction["values"][indices], budget, rtol=1e-08, atol=1e-07
        )


def option_values(option):
    return option[1]["values"] if option is not None else None


def peak_screens(data, records):
    """Assess threshold relevance separately from approval of hourly values.

    A profile can fail hourly MAE at very low discharge while its peak errors
    still support threshold classification. Use the largest validation
    underprediction for this screen, and leave failed hourly fills missing.
    """
    rows = []
    for record in records:
        positions = record["positions"]
        blocks = [
            (int(a), int(b))
            for _, group in data["components"]
            for gid, a, b in group
            if gid in record["gap_ids"]
        ]
        tests = [
            s
            for s in data["validation"]
            if s["component_id"] == record["component_id"] and s["scope"] == "component"
        ]
        supported = []
        for s in tests:
            if s["test_windows"] < 12 or s["test_years"] < 4:
                continue
            if s["median_peak_error"] > 0.1 or s["p90_peak_error"] > 0.15:
                continue
            upper = s["predicted_peak_m3_s"] + max(0.0, s["maximum_peak_residual"])
            sensitive, basis = decision_sensitive(
                boundary_context(data, blocks),
                positions,
                blocks,
                upper,
                data["threshold"],
            )
            if not sensitive:
                supported.append((methods.index(s["method"]), s, upper, basis))
        if not supported:
            continue
        _, s, upper, basis = min(supported, key=lambda x: x[0])
        rows.append(
            dict(
                component_id=record["component_id"],
                method=s["method"],
                test_windows=s["test_windows"],
                test_years=s["test_years"],
                relative_mae=s["relative_mae"],
                median_peak_error=s["median_peak_error"],
                p90_peak_error=s["p90_peak_error"],
                upper_peak_screen_m3_s=upper,
                hourly_profile_approved=s["hourly_pass"],
                screen_basis=basis,
                screen_status="validated_peak_screen",
                uncertainty="Empirical screen using maximum test underprediction, conditional on comparable periods.",
            )
        )
    return pd.DataFrame(
        rows,
        columns=[
            "component_id",
            "method",
            "test_windows",
            "test_years",
            "relative_mae",
            "median_peak_error",
            "p90_peak_error",
            "upper_peak_screen_m3_s",
            "hourly_profile_approved",
            "screen_basis",
            "screen_status",
            "uncertainty",
        ],
    )


def joint_check(data, q, candidate, records):
    """Retest complete affected event masks, including mixed selected methods."""
    by_gap = {g[0]: g for _, group in data["components"] for g in group}
    tested = set()
    for _ in range(len(records) + 1):
        changed = False
        for event in extract(candidate, data["threshold"]):
            if not event["complete"]:
                continue
            context = context_bounds(
                candidate, event["first"], event["last"] + 1, data["threshold"]
            )
            if context is None:
                continue
            left, right = context
            related = [
                r
                for r in records
                if r["candidate"] is not None
                and np.any((r["positions"] >= left) & (r["positions"] < right))
            ]
            contexts = [context]
            for _ in range(len(records) + 1):
                old_ids = {r["component_id"] for r in related}
                positions = (
                    np.concatenate([r["positions"] for r in related])
                    if related
                    else np.array([], int)
                )
                for neighbour in extract(candidate, data["threshold"]):
                    if np.any(
                        (positions >= neighbour["first"] - separation_hours)
                        & (positions <= neighbour["last"] + separation_hours)
                    ):
                        bounds = context_bounds(
                            candidate,
                            neighbour["first"],
                            neighbour["last"] + 1,
                            data["threshold"],
                        )
                        if bounds is not None:
                            contexts.append(bounds)
                related = [
                    r
                    for r in records
                    if r["candidate"] is not None
                    and any(
                        (
                            np.any((r["positions"] >= a) & (r["positions"] < b))
                            for a, b in contexts
                        )
                    )
                ]
                if old_ids == {r["component_id"] for r in related}:
                    break
            if len(related) < 2:
                continue
            ids = tuple((r["component_id"] for r in related))
            if ids in tested:
                continue
            tested.add(ids)
            group = sorted(
                [by_gap[g] for r in related for g in r["gap_ids"]], key=lambda g: g[1]
            )
            methods = {g: r["candidate"][0] for r in related for g in r["gap_ids"]}
            positions, proposed, approved, status = assess(
                data, group, fixed_methods=methods
            )
            row = dict(
                component_ids=",".join(map(str, ids)),
                status=status,
                accepted=approved is not None,
                combined_mask_hours=len(positions),
            )
            if approved is not None:
                verify_prediction(data, positions, approved)
                release = data["allowed"][positions]
                candidate[positions[release]] = approved[1]["values"][release]
                q[positions[release]] = approved[1]["values"][release]
                for r in related:
                    location = np.searchsorted(positions, r["positions"])
                    prediction = make_prediction(
                        approved[1]["values"][location], approved[1]["donor_spec"]
                    )
                    r["pot"] = (r["candidate"][0], prediction, approved[2], approved[3])
                    r["joint_group"] = ids
                changed = True
            data["joint_rows"].append(row)
        for event in extract(q, data["threshold"]):
            if not event["complete"]:
                continue
            bounds = context_bounds(
                q, event["first"], event["last"] + 1, data["threshold"]
            )
            if bounds is None:
                continue
            left, right = bounds
            related = [
                r
                for r in records
                if r["pot"] is not None
                and np.any((r["positions"] >= left) & (r["positions"] < right))
            ]
            if len(related) > 1:
                ids = tuple((r["component_id"] for r in related))
                groups = {tuple(r.get("joint_group", ())) for r in related}
                covered = len(groups) == 1 and set(ids).issubset(next(iter(groups)))
                if not covered:
                    for r in related:
                        q[r["positions"]] = np.nan
                        r["pot"] = None
                        r["release_reason"] = "combined_event_context_not_validated"
                    changed = True
        if not changed:
            break
    return (q, candidate)


def daily_days(a, b):
    return daily_starts(a, b - 1)


test_lengths = (6, 24, 72, 168)
flood_level = 20.0
matching_days = 56
context_hours = max(test_lengths) // 2 + separation_hours


def usable_context(data, center, years):
    left, right = (center - context_hours, center + context_hours + 1)
    return (
        left >= 0
        and right <= data["n"]
        and (data["year"][center] in years)
        and (data["year"][left] == data["year"][right - 1])
        and (data["missing_prefix"][left] == data["missing_prefix"][right])
    )


def spec_label(spec):
    return ", ".join(
        (f"{donor_stations[sid]} ({lag:+d} h)" for sid, lag in zip(spec[0], spec[1]))
    )


def predictions_at(data, center, audit, failures):
    """Lock feasibility without inspecting reconstruction errors."""
    year = int(data["year"][center])
    held = (year,)
    observed = data["q"]
    predictions = {}
    for hours in test_lengths:
        first, stop = (center - hours // 2, center + hours // 2)
        hidden = np.arange(first, stop)
        assert len(hidden) == hours and np.all(data["year"][hidden] == year)
        days = daily_starts(first, stop - 1)
        required_days = []
        for day in days:
            samples = np.arange(day, day + 25)
            weights = np.r_[0.5, np.ones(23), 0.5]
            if weights[np.isin(samples, hidden)].sum() > 0.5:
                required_days.append(int(day))
        if any((day not in data["daily"] for day in required_days)):
            failures["incomplete_daily_support"] += 1
            return None
        data["q"] = observed.copy()
        data["q"][hidden] = np.nan
        try:
            fitted = predict(data, hidden, [(first, stop)], held)
        finally:
            data["q"] = observed
        if set(fitted) != set(methods):
            failures["missing_method_or_nonnegative_feasibility"] += 1
            return None
        for method, pred in fitted.items():
            assert len(pred["values"]) == hours
            assert np.isfinite(pred["values"]).all() and np.all(pred["values"] >= 0)
            if method.startswith("daily_"):
                assert {int(c[0]) for c in pred["constraints"]} == set(required_days)
            max_residual = 0.0
            for day, flag, indices, weights, budget in pred["constraints"]:
                assert str(flag) in {"0", "1"}
                residual = float(abs(weights @ pred["values"][indices] - budget))
                max_residual = max(max_residual, residual)
                assert np.isclose(
                    weights @ pred["values"][indices], budget, rtol=1e-08, atol=1e-07
                )
            train_count = 0
            if pred["donor_spec"]:
                spec = next(
                    (
                        s
                        for s in donor_ranking(data, held)
                        if spec_label(s) == pred["donor_spec"]
                    )
                )
                xx, xy, yy, count = regression_sums(data["models"], spec)
                allowed = ~np.isin(data["models"]["years"], held)
                assert not np.isin(data["models"]["years"][allowed], held).any()
                train_count = int(count[allowed].sum())
                assert train_count >= 720
            audit.append(
                dict(
                    center_utc=data["index"][center],
                    year=year,
                    held_year=year,
                    hours=hours,
                    method=method,
                    donor_spec=pred["donor_spec"],
                    train_rows=train_count,
                    hidden_target_rows_in_training=0,
                    cross_year_lag_rows_excluded=True,
                    daily_constraints=len(pred["constraints"]),
                    max_daily_constraint_residual=max_residual,
                    complete_daily_support=True,
                )
            )
        predictions[hours] = (hidden, fitted)
    return predictions


def build_cases(data, years):
    flood_events = [
        e
        for e in extract(data["q"], flood_level)
        if e["complete"] and usable_context(data, e["peak"], years)
    ]
    audit, pairs, used_quiet, failures = ([], [], [], Counter())
    offsets = sorted(
        range(-matching_days, matching_days + 1), key=lambda x: (abs(x), x)
    )
    for event in flood_events:
        center = event["peak"]
        flood = predictions_at(data, center, audit, failures)
        if flood is None:
            continue
        for days in offsets:
            quiet = center + 24 * days
            if not usable_context(data, quiet, years):
                continue
            if data["year"][quiet] != data["year"][center]:
                continue
            if any(
                (abs(quiet - other) < 2 * context_hours + 1 for other in used_quiet)
            ):
                continue
            if (
                data["q"][quiet - context_hours : quiet + context_hours + 1].max()
                > flood_level
            ):
                continue
            quiet_fit = predictions_at(data, quiet, audit, failures)
            if quiet_fit is None:
                continue
            pairs.append(((center, flood), (quiet, quiet_fit)))
            used_quiet.append(quiet)
            print(
                f"Pair {len(pairs):2}: flood {data['index'][center]} / quiet {data['index'][quiet]}",
                flush=True,
            )
            break
        else:
            failures["no_comparable_feasible_quiet_window"] += 1
    return (pairs, audit, failures, len(flood_events))


def score_cases(data, pairs):
    records, cases, profiles = ([], [], [])
    for pair_id, pair in enumerate(pairs, 1):
        flood_year = int(data["year"][pair[0][0]])
        assert data["year"][pair[1][0]] == flood_year
        assert abs(pair[1][0] - pair[0][0]) <= 24 * matching_days
        for regime, (center, fits) in zip(("Flood", "Non-flood"), pair):
            cases.append(
                dict(
                    pair=pair_id,
                    regime=regime,
                    year=flood_year,
                    center_utc=data["index"][center],
                    max_context_m3_s=float(
                        data["q"][
                            center - context_hours : center + context_hours + 1
                        ].max()
                    ),
                )
            )
            for hours, (hidden, predictions) in fits.items():
                truth = data["q"][hidden]
                assert len(hidden) == hours and (truth.max() > flood_level) == (
                    regime == "Flood"
                )
                for method, pred in predictions.items():
                    bias = float(pred["values"].max() - truth.max())
                    error = np.abs(pred["values"] - truth)
                    records.append(
                        dict(
                            pair=pair_id,
                            regime=regime,
                            year=flood_year,
                            center_utc=data["index"][center],
                            hours=hours,
                            method=method,
                            measured_peak_m3_s=float(truth.max()),
                            estimated_peak_m3_s=float(pred["values"].max()),
                            peak_error_m3_s=abs(bias),
                            peak_error_pct=100 * abs(bias) / truth.max(),
                            peak_bias_m3_s=bias,
                            peak_bias_pct=100 * bias / truth.max(),
                            mae_m3_s=float(error.mean()),
                            mae_pct=100 * float(error.mean()) / truth.mean(),
                            timing_error_h=int(
                                abs(pred["values"].argmax() - truth.argmax())
                            ),
                            donor_spec=pred["donor_spec"],
                        )
                    )
                    if (
                        regime == "Flood"
                        and data["index"][center]
                        == pd.Timestamp("2018-05-11 21:00", tz="UTC")
                        and (hours == 24)
                    ):
                        for i, v in zip(hidden, pred["values"]):
                            profiles.append(
                                dict(
                                    timestamp_utc=data["index"][i],
                                    method=method,
                                    estimate_m3_s=float(v),
                                    observed_m3_s=float(data["q"][i]),
                                )
                            )
    records = pd.DataFrame(records)
    expected = set(range(1, len(pairs) + 1))
    for _, rows in records.groupby(["regime", "hours", "method"]):
        assert set(rows.pair) == expected and len(rows) == len(expected)
    assert len(records) == len(pairs) * 2 * len(test_lengths) * len(methods)
    return (records, pd.DataFrame(cases), pd.DataFrame(profiles))


def run_matched_tests(years):
    """Recreate the four-method comparison without changing gap approvals."""
    data = create_reconstruction()
    pairs, audit, failures, candidates = build_cases(data, set(years))
    errors, cases, profiles = score_cases(data, pairs)
    folder = output_folder / "benchmark"
    folder.mkdir(parents=True, exist_ok=True)
    errors.to_csv(folder / "matched_benchmark_errors.csv", index=False)
    cases.to_csv(folder / "matched_cases.csv", index=False)
    profiles.to_csv(folder / "may_2018_predictions.csv", index=False)
    first = data["index"].get_loc(pd.Timestamp("2018-05-10 03:00", tz="UTC"))
    stop = data["index"].get_loc(pd.Timestamp("2018-05-14 03:00", tz="UTC"))
    pd.DataFrame(
        {
            "timestamp_utc": data["index"][first : stop + 1],
            "observed_m3_s": data["q"][first : stop + 1],
        }
    ).to_csv(folder / "may_2018_observations.csv", index=False)

    pd.DataFrame(audit).to_csv(folder / "benchmark_audit.csv", index=False)
    return (errors, cases, profiles)


def export_reconstruction(data, records):
    candidate, main = (data["q"].copy(), data["q"].copy())
    for r in records:
        eligible = data["allowed"][r["positions"]]
        if r["candidate"] is not None:
            candidate[r["positions"][eligible]] = r["candidate"][1]["values"][eligible]
        if r["pot"] is not None:
            main[r["positions"][eligible]] = r["pot"][1]["values"][eligible]
    data["joint_rows"] = []
    main, candidate = joint_check(data, main, candidate, records)
    gap_id, component = (np.full(data["n"], np.nan), np.full(data["n"], np.nan))
    main_method, candidate_method = (
        np.full(data["n"], "", object),
        np.full(data["n"], "", object),
    )
    factors = np.full((data["n"], 2), np.nan)
    unresolved = np.zeros(data["n"], bool)
    gap_rows = []
    for r in records:
        positions = r["positions"]
        component[positions] = r["component_id"]
        c, p = (r["candidate"], r["pot"])
        reason = r.get("release_reason") or (
            p[3]["reason"] if p else c[3]["reason"] if c else r["status"]
        )
        if c is not None:
            candidate_method[positions] = c[0]
        if p is not None:
            main_method[positions] = p[0]
            factors[positions] = (p[2]["factor_p05"], p[2]["factor_p95"])
        else:
            unresolved[positions] = True
        for row in (
            data["gaps"].loc[data["gaps"].gap_id.isin(r["gap_ids"])].itertuples()
        ):
            part = np.arange(row.a, row.b)
            gap_id[part] = row.gap_id
            new_n = int(np.isfinite(main[part]).sum())
            candidate_n = int(np.isfinite(candidate[part]).sum())
            gap_rows.append(
                dict(
                    gap_id=row.gap_id,
                    component_id=r["component_id"],
                    start_utc=row.start_utc,
                    end_utc=row.end_utc,
                    missing_hours=row.missing_hours,
                    pot_reconstructed_hours=new_n,
                    candidate_reconstructed_hours=candidate_n,
                    remaining_hours=len(part) - new_n,
                    candidate_method=c[0] if c else "",
                    pot_method=p[0] if p else "",
                    pot_approved=p is not None,
                    release_reason=reason,
                    input_daily_codes=",".join(
                        sorted(
                            {
                                data["daily"][d][1]
                                for d in daily_days(row.a, row.b)
                                if d in data["daily"]
                            }
                        )
                    ),
                    source_pattern_start_utc=row.source_pattern_start_utc,
                    source_pattern_end_utc=row.source_pattern_end_utc,
                    source_pattern_hours=row.source_pattern_hours,
                    validation_pattern_hours=len(positions),
                    boundary_anchors_utc=",".join(
                        (
                            str(data["index"][i])
                            for i in sorted(data["boundary_anchors"])
                            if i == int(positions[0]) - 1 or i == int(positions[-1]) + 1
                        )
                    ),
                )
            )
    observed = np.isfinite(data["q"])
    assert np.array_equal(main[observed], data["q"][observed])
    assert np.array_equal(candidate[observed], data["q"][observed])
    assert np.all(main[np.isfinite(main)] >= 0) and np.all(
        candidate[np.isfinite(candidate)] >= 0
    )
    assert not np.isfinite(main[~data["allowed"]]).any()
    assert not np.isfinite(candidate[~data["allowed"]]).any()
    main_method[~data["allowed"]] = ""
    candidate_method[~data["allowed"]] = ""
    gaps = pd.DataFrame(gap_rows)
    screens = peak_screens(data, records)
    screen_map = screens.set_index("component_id").to_dict("index")
    lower_context, upper_context = (main.copy(), main.copy())
    for record in records:
        if record["component_id"] not in screen_map:
            continue
        positions = record["positions"]
        unknown = positions[data["allowed"][positions] & ~np.isfinite(main[positions])]
        lower_context[unknown] = 0.0
        upper_context[unknown] = screen_map[record["component_id"]][
            "upper_peak_screen_m3_s"
        ]
    event_signature = lambda q: [
        (e["first"], e["last"], e["peak"], e["complete"])
        for e in extract(q, data["threshold"])
    ]
    assert event_signature(lower_context) == event_signature(upper_context)
    gaps["flood_risk_screen_passed"] = gaps.component_id.isin(screen_map)
    year_rows = []
    for year in range(1987, 2026):
        mask = data["year"] == year
        part = main[mask]
        longest = max((b - a for a, b in runs(~np.isfinite(part))), default=0)
        open_gaps = gaps[
            (pd.to_datetime(gaps.start_utc, utc=True).dt.year <= year)
            & (pd.to_datetime(gaps.end_utc, utc=True).dt.year >= year)
            & gaps.remaining_hours.gt(0)
        ]
        unresolved_risk = open_gaps[~open_gaps.flood_risk_screen_passed]
        unknown_events = [
            e
            for e in extract(upper_context, data["threshold"])
            if not e["complete"] and data["year"][e["peak"]] == year
        ]
        eligible = (
            year not in excluded_years
            and longest <= 120
            and (len(unresolved_risk) == 0)
            and (not unknown_events)
        )
        reason = (
            "nve_excluded"
            if year in excluded_years
            else (
                "remaining_gap_above_120_hours"
                if longest > 120
                else (
                    "unresolved_gap_flood_risk"
                    if len(unresolved_risk)
                    else (
                        "incomplete_event_context"
                        if unknown_events
                        else "passed_coverage_and_event_checks"
                    )
                )
            )
        )
        year_rows.append(
            dict(
                year=year,
                calendar_hours=int(mask.sum()),
                observed_hours=int(np.isfinite(data["q"][mask]).sum()),
                reconstructed_hours=int((~observed[mask] & np.isfinite(part)).sum()),
                candidate_reconstructed_hours=int(
                    (~observed[mask] & np.isfinite(candidate[mask])).sum()
                ),
                remaining_hours=int((~np.isfinite(part)).sum()),
                longest_remaining_gap_hours=longest,
                unresolved_gap_ids=",".join(map(str, open_gaps.gap_id)),
                unresolved_flood_risk_gap_ids=",".join(
                    map(str, unresolved_risk.gap_id)
                ),
                incomplete_event_candidates=len(unknown_events),
                included_after=eligible,
                reason=reason,
                baseline_year=year in reference_years,
            )
        )
    years = pd.DataFrame(year_rows)
    conflicts = []
    for r in records:
        affected_years = set(data["year"][r["positions"]]) & set(reference_years)
        if not affected_years:
            continue
        c = r["candidate"]
        safe = r["component_id"] in screen_map
        if not safe:
            tests = [
                x
                for x in data["validation"]
                if x["component_id"] == r["component_id"] and x["scope"] == "component"
            ]
            enough = any(
                (x["test_windows"] >= 12 and x["test_years"] >= 4 for x in tests)
            )
            basis = (
                "insufficient_comparable_test_windows"
                if not enough
                else (
                    "no_method_passed_hourly_error_requirements"
                    if c is None
                    else c[3].get("risk_screen_basis", "event_effect_requires_review")
                )
            )
            conflicts.append(
                dict(
                    component_id=r["component_id"],
                    years=",".join(map(str, sorted(affected_years))),
                    gap_ids=",".join(map(str, r["gap_ids"])),
                    reason="review_required_not_a_failed_year_verdict",
                    review_basis=basis,
                    candidate_reason=c[3]["reason"] if c else r["status"],
                )
            )
    hourly = pd.DataFrame(
        dict(
            timestamp_utc=data["index"],
            year=data["year"],
            source_discharge_m3_s=data["raw"],
            source_correction=data["hourly"].Korrigert.to_numpy(),
            source_quality_control=data["hourly"].Kontrollert.to_numpy(),
            observed_discharge_m3_s=data["q"],
            candidate_discharge_m3_s=candidate,
            pot_discharge_m3_s=main,
            gap_id=gap_id,
            component_id=component,
            event_context_upper_m3_s=upper_context,
            event_context_lower_m3_s=lower_context,
            candidate_method=candidate_method,
            pot_method=main_method,
            peak_factor_p05=factors[:, 0],
            peak_factor_p95=factors[:, 1],
            baseline_year=np.isin(data["year"], reference_years),
            after_year_included=np.isin(
                data["year"], years.loc[years.included_after, "year"]
            ),
        )
    )
    hourly.to_csv(reconstruction_folder / "hourly_reconstruction.csv", index=False)
    gaps.to_csv(reconstruction_folder / "gap_decisions.csv", index=False)
    screens.to_csv(reconstruction_folder / "gap_flood_risk_screens.csv", index=False)
    years.to_csv(reconstruction_folder / "year_summary.csv", index=False)
    pd.DataFrame(
        conflicts,
        columns=[
            "component_id",
            "years",
            "gap_ids",
            "reason",
            "review_basis",
            "candidate_reason",
        ],
    ).to_csv(reconstruction_folder / "baseline_review.csv", index=False)
    pd.DataFrame(data["validation"]).to_csv(
        reconstruction_folder / "validation_scores.csv", index=False
    )
    pd.DataFrame(data["window_results"]).to_csv(
        reconstruction_folder / "validation_windows.csv", index=False
    )
    pd.DataFrame(data["training_audit"]).to_csv(
        reconstruction_folder / "training_audit.csv", index=False
    )
    pd.DataFrame(
        [
            dict(
                timestamp_utc=data["index"][i],
                discharge_m3_s=value,
                correction=int(data["hourly"].Korrigert.iloc[i]),
                quality_control=int(data["hourly"].Kontrollert.iloc[i]),
                role="boundary_only_never_training_or_released_year",
            )
            for i, value in sorted(data["boundary_anchors"].items())
        ]
    ).to_csv(reconstruction_folder / "boundary_anchors.csv", index=False)
    pd.DataFrame(
        data["joint_rows"],
        columns=["component_ids", "status", "accepted", "combined_mask_hours"],
    ).to_csv(reconstruction_folder / "combined_context_checks.csv", index=False)
    monthly = (
        hourly.assign(month=data["index"].month)
        .groupby(["year", "month"])
        .agg(
            calendar_hours=("year", "size"),
            observed_hours=("observed_discharge_m3_s", "count"),
            candidate_available_hours=("candidate_discharge_m3_s", "count"),
            pot_available_hours=("pot_discharge_m3_s", "count"),
        )
        .reset_index()
    )
    monthly.to_csv(reconstruction_folder / "monthly_coverage.csv", index=False)
    assert monthly.observed_hours.sum() == observed.sum()
    assert monthly.pot_available_hours.sum() == np.isfinite(main).sum()
    constraints_checked = 0
    for r in records:
        for label, option, series in [
            ("candidate", r["candidate"], candidate),
            ("pot", r["pot"], main),
        ]:
            if option is None or r.get("joint_group"):
                continue
            for day, _, indices, w, budget in option[1]["constraints"]:
                values = series[r["positions"]].copy()
                context_only = ~data["allowed"][r["positions"]]
                values[context_only] = option[1]["values"][context_only]
                assert np.isclose(w @ values[indices], budget, rtol=1e-08, atol=1e-07)
                constraints_checked += 1
    hourly.index = pd.to_datetime(hourly.timestamp_utc, utc=True)
    return (hourly, years, pd.DataFrame(data["validation"]))


def reconstruct_all():
    """Recompute gap tests and approvals from the original discharge exports."""
    reconstruction_folder.mkdir(parents=True, exist_ok=True)
    data = create_reconstruction()
    records = []
    total = len(data["components"])
    for number, (_, group) in enumerate(data["components"], 1):
        ids = [gap_id for gap_id, _, _ in group]
        source_width = (
            data["gaps"]
            .loc[data["gaps"].gap_id.isin(ids), "source_pattern_hours"]
            .max()
        )
        if (
            max((stop - first for _, first, stop in group)) > maximum_gap
            or source_width > maximum_gap
        ):
            positions = np.concatenate(
                [np.arange(first, stop) for _, first, stop in group]
            )
            candidate, approved, status = (None, None, "gap_above_1050_hours")
        else:
            positions, candidate, approved, status = assess(data, group)
        verify_prediction(data, positions, candidate)
        verify_prediction(data, positions, approved)
        records.append(
            dict(
                component_id=group[0][0],
                gap_ids=ids,
                positions=positions,
                candidate=candidate,
                pot=approved,
                status=status,
            )
        )
        if number == 1 or number % 10 == 0 or number == total:
            print(f"Validated {number}/{total} gap components", flush=True)
    return export_reconstruction(data, records)
