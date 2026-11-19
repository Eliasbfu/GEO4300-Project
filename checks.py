"""Numerical checks for the event rules, reconstruction and EVA results."""

import numpy as np
import pandas as pd

import reconstruction


def check_event_rules():
    values = np.zeros(1800)
    values[400], values[712], values[1100] = 12, 15, 10
    assert len(reconstruction.extract(values, 10)) == 1
    values[712], values[713] = 0, 15
    assert len(reconstruction.extract(values, 10)) == 2

    values = np.full(1800, 10.0)
    values[400], values[800] = 15, 30
    assert len(reconstruction.extract(values, 11)) == 1
    values[600] = 9.9
    assert len(reconstruction.extract(values, 11)) == 2
    values[600] = np.nan
    assert not any(event["complete"] for event in reconstruction.extract(values, 11))

    observed = np.zeros(2500)
    observed[600:650], observed[1100:1150] = 20, 25
    merged = observed.copy()
    merged[650:1100] = 19
    score = reconstruction.compare(observed, merged, np.arange(650, 1100), 10)
    assert len(score["matches"]) == 1 and len(score["missed"]) == 1

    values = np.r_[np.ones(8760), np.full(3 * 8760, np.nan), np.ones(8760)]
    np.testing.assert_allclose(
        reconstruction.exposure_years(values), 2 * 8760 / (24 * 365.2425)
    )


def check_daily_adjustment():
    positions = np.array([22, 23, 24])
    constraints = [
        (-1, "0", np.array([0, 1]), np.array([1.0, 0.5]), 2.0),
        (23, "0", np.array([1, 2]), np.array([0.5, 1.0]), 2.0),
    ]
    fit = reconstruction.daily_adjust(np.ones(3), positions, constraints, np.zeros(3))
    np.testing.assert_allclose(fit, [1.3, 1.4, 1.3], atol=1e-10)

    data = dict(
        q=np.array([np.nan, np.nan, np.nan, np.nan, 3.0]),
        n=5,
        boundary_anchors={0: 1.0},
        daily={},
        right_anchors={},
    )
    original = data["q"].copy()
    prediction = reconstruction.predict(
        data, np.arange(1, 4), [(1, 4)], (), methods=("linear",)
    )
    np.testing.assert_allclose(prediction["linear"]["values"], [1.5, 2.0, 2.5])
    np.testing.assert_equal(data["q"], original)


def check_record(hourly, years):
    observed = hourly.observed_discharge_m3_s.to_numpy(float)
    reconstructed = hourly.pot_discharge_m3_s.to_numpy(float)
    known = np.isfinite(observed)
    np.testing.assert_array_equal(observed[known], reconstructed[known])
    assert np.all(reconstructed[np.isfinite(reconstructed)] >= 0)
    assert not np.isfinite(
        reconstructed[hourly.year.isin(reconstruction.excluded_years)]
    ).any()
    assert int(years.reconstructed_hours.sum()) == 20077
    assert years.included_after.sum() == 28
    included = years.loc[years.included_after]
    assert included.remaining_hours.sum() == 120
    assert (included.remaining_hours == 0).sum() == 27
    assert included.longest_remaining_gap_hours.max() <= 120


def check_benchmark(errors):
    assert errors.pair.nunique() == 13
    for _, group in errors.groupby(["regime", "hours", "method"]):
        assert set(group.pair) == set(range(1, 14))
    assert len(errors) == 13 * 2 * 4 * 4
    assert set(errors.hours) == {6, 24, 72, 168}


def check_eva(summary, levels, events, bootstraps):
    reference = pd.read_csv(
        reconstruction.reference_folder / "eva_summary.csv"
    ).set_index("sample")
    current = summary.set_index("sample")
    fields = [
        "events",
        "exposure_years",
        "events_per_exposure_year",
        "mean_peak_m3_s",
        "sample_sd_peak_m3_s",
        "skewness",
        "gpd_shape",
        "gpd_scale",
    ]
    np.testing.assert_allclose(
        current.loc[["before", "after"], fields],
        reference.loc[["before", "after"], fields],
        rtol=1e-10,
        atol=1e-10,
    )
    expected = pd.read_csv(reconstruction.reference_folder / "return_levels.csv")
    expected = expected.loc[expected["sample"].isin(["before", "after"])]
    fields = ["return_level_m3_s", "lower_95_m3_s", "upper_95_m3_s"]
    np.testing.assert_allclose(levels[fields], expected[fields], rtol=1e-9, atol=1e-9)
    assert not events.event_source.eq("estimated_peak").any()
    assert (
        events.loc[events["sample"].eq("before"), "event_source"]
        .eq("measured_peak_with_validated_context")
        .sum()
        == 4
    )
    assert (
        events.loc[events["sample"].eq("after"), "event_source"]
        .eq("measured_peak_with_validated_context")
        .sum()
        == 7
    )
    for name, draws in bootstraps.items():
        saved = pd.read_csv(reconstruction.reference_folder / f"{name}_bootstrap.csv")
        np.testing.assert_allclose(draws, saved[draws.columns], rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    check_event_rules()
    check_daily_adjustment()
    hourly, years, scores = reconstruction.read_reconstruction()
    check_record(hourly, years)
    check_benchmark(
        pd.read_csv(
            reconstruction.saved_folder / "benchmark/matched_benchmark_errors.csv"
        )
    )
    print("Event, daily-constraint, record and matched-case checks passed.")
