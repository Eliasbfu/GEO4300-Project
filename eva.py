"""Event samples, exposure, GPD fitting and bootstrap return levels."""

import numpy as np
import pandas as pd
from scipy import stats
import reconstruction


def reference_context(hourly, screens, threshold):
    """Use validated gap context without filling reference observations."""
    observed = hourly.observed_discharge_m3_s.to_numpy(float)
    context, lower, upper = (observed.copy(), observed.copy(), observed.copy())
    screens = screens.set_index("component_id")
    missing = hourly.baseline_year & hourly.observed_discharge_m3_s.isna()
    for component, part in hourly.loc[missing].groupby("component_id"):
        positions = hourly.index.get_indexer(part.index)
        ceiling = float(screens.loc[component, "upper_peak_screen_m3_s"])
        estimates = part.pot_discharge_m3_s.to_numpy(float)
        if np.isfinite(estimates).any():
            ceiling = max(ceiling, float(np.nanmax(estimates)))
        assert ceiling < threshold
        lower[positions], upper[positions] = (0.0, ceiling)
        context[positions] = np.where(np.isfinite(estimates), estimates, ceiling)
    selected = hourly.baseline_year.to_numpy(bool)
    signatures = []
    for values in (lower, upper, context):
        events = reconstruction.extract(values, threshold)
        signatures.append(
            [
                (e["first"], e["last"], e["peak"], e["complete"])
                for e in events
                if selected[e["peak"]]
            ]
        )
    assert signatures[0] == signatures[1] == signatures[2]
    return context


def select_events(hourly, years, column, threshold, context):
    """Select peaks by their UTC year and count only available-hour exposure."""
    selected = hourly.year.isin(years).to_numpy()
    discharge = hourly[column].to_numpy(float)
    observed = hourly.observed_discharge_m3_s.to_numpy(float)
    measured_events = reconstruction.extract(observed, threshold)
    complete_measured = {
        (e["first"], e["last"], e["peak"]) for e in measured_events if e["complete"]
    }
    rows = []
    for event in reconstruction.extract(context, threshold):
        peak = event["peak"]
        if not selected[peak]:
            continue
        assert event["complete"], "An event still needs a gap-context assessment."
        assert np.isfinite(discharge[peak])
        key = (event["first"], event["last"], peak)
        if not np.isfinite(observed[peak]):
            source = "estimated_peak"
        elif key in complete_measured:
            source = "fully_observed"
        else:
            source = "measured_peak_with_validated_context"
        rows.append(
            dict(
                peak_utc=hourly.index[peak],
                peak_year=int(hourly.year.iloc[peak]),
                peak_m3_s=discharge[peak],
                event_source=source,
                first_exceedance_utc=hourly.index[event["first"]],
                last_exceedance_utc=hourly.index[event["last"]],
            )
        )
    exposure = reconstruction.exposure_years(np.where(selected, discharge, np.nan))
    return (pd.DataFrame(rows), exposure)


def return_levels(threshold, shape, scale, rate, periods=(20, 200, 1000)):
    """Invert the GPD survivor function using the annual event rate."""
    periods = np.asarray(periods, float)
    if abs(shape) < 1e-08:
        return threshold + scale * np.log(rate * periods)
    return threshold + scale / shape * ((rate * periods) ** shape - 1)


def fit_sample(events, exposure, threshold, seed=4300, samples=1000):
    """Fit a GPD and resample the event peaks with a fixed annual rate."""
    peaks = events.peak_m3_s.to_numpy(float)
    excess = peaks - threshold
    assert np.all(excess > 0) and exposure > 0
    shape, _, scale = stats.genpareto.fit(excess, floc=0)
    rate = len(peaks) / exposure
    periods = np.array([20, 200, 1000])
    levels = return_levels(threshold, shape, scale, rate, periods)
    independent = threshold + stats.genpareto.isf(
        1 / (rate * periods), shape, loc=0, scale=scale
    )
    np.testing.assert_allclose(levels, independent, rtol=1e-09, atol=1e-09)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**32 - 1, size=samples, dtype=np.uint32)
    draws = []
    for draw_seed in seeds:
        sample = np.random.default_rng(int(draw_seed)).choice(
            excess, size=len(excess), replace=True
        )
        draw_shape, _, draw_scale = stats.genpareto.fit(sample, floc=0)
        assert (
            draw_scale > 0
            and np.isfinite(
                stats.genpareto.logpdf(sample, draw_shape, scale=draw_scale)
            ).all()
        )
        draws.append(return_levels(threshold, draw_shape, draw_scale, rate, periods))
    draws = np.asarray(draws)
    assert np.isfinite(draws).all()
    lower, upper = np.quantile(draws, [0.025, 0.975], axis=0)
    summary = dict(
        events=len(peaks),
        exposure_years=exposure,
        events_per_exposure_year=rate,
        threshold_m3_s=threshold,
        mean_peak_m3_s=peaks.mean(),
        sample_sd_peak_m3_s=peaks.std(ddof=1),
        skewness=stats.skew(peaks),
        gpd_shape=shape,
        gpd_scale=scale,
    )
    levels = pd.DataFrame(
        dict(
            return_period_years=periods,
            return_level_m3_s=levels,
            lower_95_m3_s=lower,
            upper_95_m3_s=upper,
        )
    )
    draws = pd.DataFrame(draws, columns=["Q20_m3_s", "Q200_m3_s", "Q1000_m3_s"])
    return (summary, levels, draws)
