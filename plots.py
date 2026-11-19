"""Figures for the discharge reconstruction and flood-frequency analysis."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as patheffects
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection

folder = Path(__file__).resolve().parent
figures = folder / "Figures"
benchmark_folder = folder / "data/processed/benchmark"
map_folder = folder / "data/map"
ink = "#171a1d"
frame_color = "#869099"
before_color = "#355f8d"
after_color = "#16806a"
filled_color = "#b8672b"
gray = "#b8bcc1"
method_styles = {
    "linear": ("#b8672b", (0, (2, 1.3)), "Linear interpolation"),
    "daily_linear": ("#355f8d", (0, (5, 1.4)), "Interpolation + daily mean"),
    "daily_donor_ols": ("#16806a", (0, (4, 1.2, 1, 1.2)), "Regression + daily mean"),
}
methods = ("linear", "daily_linear", "donor_ols", "daily_donor_ols")
map_bounds = (10.702696301832887, 12.797303698167113, 61.83, 62.77)
radius = 6378137.0
earth_km = 6371.0088
target_color = "#a93535"
donor_color = "#187c7d"


def set_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8.2,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.65,
            "text.color": ink,
            "axes.labelcolor": ink,
            "xtick.color": ink,
            "ytick.color": ink,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def hydrograph(fig, ax, legend_y, directory=benchmark_folder):
    observed = pd.read_csv(
        directory / "may_2018_observations.csv", parse_dates=["timestamp_utc"]
    )
    observed = observed.set_index("timestamp_utc").observed_m3_s
    profiles = pd.read_csv(
        directory / "may_2018_predictions.csv", parse_dates=["timestamp_utc"]
    )
    first, stop = (pd.Timestamp("2018-05-11T09:00Z"), pd.Timestamp("2018-05-12T09:00Z"))
    ax.axvspan(first, stop, color="#e9eff3", lw=0, zorder=0)
    ax.plot(observed.index, observed, color="#202020", lw=1.25, zorder=2)
    for method, (color, dash, label) in method_styles.items():
        curve = (
            profiles.loc[profiles.method.eq(method)]
            .set_index("timestamp_utc")
            .estimate_m3_s
        )
        stamps = [first - pd.Timedelta("1h"), *curve.index, stop]
        values = np.r_[
            observed.loc[first - pd.Timedelta("1h")], curve.values, observed.loc[stop]
        ]
        ax.plot(stamps, values, color=color, ls=dash, lw=1.25, zorder=3)
        if method != "linear":
            k = curve.idxmax()
            ax.plot(
                k,
                curve.loc[k],
                marker="o" if method == "daily_linear" else "^",
                color=color,
                ms=2.8,
                markerfacecolor="white",
                mew=0.65,
                zorder=4,
            )
    truth = observed.loc[first : stop - pd.Timedelta("1h")]
    ax.plot(truth.idxmax(), truth.max(), "o", color="#202020", ms=2.5, zorder=5)
    ax.set(
        xlim=(pd.Timestamp("2018-05-10T18:00Z"), pd.Timestamp("2018-05-12T18:00Z")),
        ylim=(24.2, 36.4),
        ylabel="Discharge ($\\mathrm{m^3\\,s^{-1}}$)",
        xlabel="Date and time (UTC)",
    )
    ax.set_yticks([25, 30, 35])
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12], tz="UTC"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M", tz="UTC"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6, tz="UTC"))
    ax.tick_params(which="major", direction="out", length=3, width=0.6, pad=4)
    ax.tick_params(which="minor", length=1.8, width=0.5)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e3e8ed", lw=0.55)
    for spine in ax.spines.values():
        spine.set_color(frame_color)
    ax.text(
        first + pd.Timedelta("12h"),
        24.6,
        "24 h withheld",
        fontsize=7.3,
        ha="center",
        va="bottom",
        color=ink,
    )
    handles = [Line2D([], [], color="#202020", lw=1.25, label="Measured discharge")]
    for method, (color, dash, label) in method_styles.items():
        handles.append(Line2D([], [], color=color, ls=dash, lw=1.25, label=label))
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.06, legend_y),
        ncol=2,
        frameon=False,
        columnspacing=1.05,
        labelspacing=0.85,
        handlelength=2.1,
        handletextpad=0.5,
        borderaxespad=0,
    )


def numeric(value):
    return f"{value:.1f}" if value >= 10 else f"{value:.2f}"


def heatmap(fig, ax, cax, records):
    grouped = records.groupby(["regime", "hours", "method"]).peak_error_pct
    quantiles = grouped.quantile([0.25, 0.5, 0.75]).unstack()
    norm = LogNorm(vmin=0.1, vmax=50)
    cmap = LinearSegmentedColormap.from_list(
        "peak_errors", ["#f7f9fb", "#c8dce9", "#315d82"]
    )
    ax.set(xlim=(0, 4.62), ylim=(11.05, 0))
    ax.set_axis_off()
    width = 0.62
    ax.text(width / 2, 0.94, "Gap", ha="center", va="center", fontsize=7.8, color=ink)
    ax.text(
        width + 1,
        0.2,
        "Interpolation",
        ha="center",
        va="center",
        fontsize=8.2,
        color=ink,
        fontweight="normal",
    )
    ax.text(
        width + 3,
        0.2,
        "Donor regression",
        ha="center",
        va="center",
        fontsize=8.2,
        color=ink,
        fontweight="normal",
    )
    for j, header in enumerate(("Without\ndaily mean", "With\ndaily mean") * 2):
        ax.text(
            width + j + 0.5,
            0.94,
            header,
            ha="center",
            va="center",
            fontsize=7.5,
            linespacing=1.1,
            color=ink,
            fontweight="normal",
        )
    for regime, top in [("Flood", 1.5), ("Non-flood", 6.25)]:
        ax.hlines(top, 0, 4.62, color="#cbd5dd", lw=0.6)
        ax.text(0.04, top + 0.31, regime, va="center", fontsize=8.5, color=ink)
        for row, hours in enumerate((6, 24, 72, 168)):
            y = top + 0.63 + row
            ax.text(
                width / 2,
                y + 0.46,
                f"{hours} h",
                ha="center",
                va="center",
                fontsize=8.0,
            )
            for column, method in enumerate(methods):
                x = width + column
                low, median, high = quantiles.loc[
                    (regime, hours, method), [0.25, 0.5, 0.75]
                ]
                color = cmap(norm(median))
                brightness = np.dot(color[:3], [0.2126, 0.7152, 0.0722])
                cell_ink = "white" if brightness < 0.58 else ink
                ax.add_patch(
                    Rectangle(
                        (x + 0.025, y + 0.025),
                        0.95,
                        0.9,
                        facecolor=color,
                        edgecolor="none",
                    )
                )
                ax.text(
                    x + 0.5,
                    y + 0.32,
                    numeric(median),
                    ha="center",
                    va="center",
                    fontsize=8.6,
                    color=cell_ink,
                    fontweight="normal",
                )
                ax.text(
                    x + 0.5,
                    y + 0.69,
                    f"{numeric(low)}–{numeric(high)}",
                    ha="center",
                    va="center",
                    fontsize=7.1,
                    color=cell_ink,
                    fontweight="normal",
                )
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="horizontal",
        ticks=[0.1, 1, 10, 50],
        format="%g",
    )
    colorbar.ax.minorticks_off()
    colorbar.ax.tick_params(labelsize=7.3, length=2.5, pad=2)
    colorbar.outline.set_edgecolor(frame_color)
    colorbar.outline.set_linewidth(0.6)
    colorbar.set_label("Absolute relative peak error (%)", fontsize=8.2, labelpad=4)


def save(fig, name):
    figures.mkdir(exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(figures / f"{name}.{extension}", dpi=400, facecolor="white")
    plt.close(fig)


def plot_reconstruction(directory=benchmark_folder):
    set_style()
    records = pd.read_csv(directory / "matched_benchmark_errors.csv")
    fig = plt.figure(figsize=(3.55, 6.95))
    fig.text(0.06, 0.981, "(a) Held-out flood, May 2018", va="top", fontsize=9.1)
    ax = fig.add_axes([0.18, 0.645, 0.8, 0.225])
    hydrograph(fig, ax, 0.945, directory)
    fig.text(0.06, 0.533, "(b) Peak errors in matched gaps", va="top", fontsize=9.1)
    matrix = fig.add_axes([0.06, 0.099, 0.92, 0.409])
    scale = fig.add_axes([0.24, 0.065, 0.56, 0.013])
    heatmap(fig, matrix, scale, records)
    save(fig, "discharge_reconstruction")


def plot_flood_frequency(hourly, summary, levels, reference, expanded):
    figures.mkdir(exist_ok=True)
    summary = summary.set_index("sample")
    levels = levels.set_index(["sample", "return_period_years"])
    reference, expanded = (set(reference), set(expanded))
    u = summary.loc["before", "threshold_m3_s"]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelsize": 9,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.65,
            "text.color": ink,
            "axes.labelcolor": ink,
            "xtick.color": ink,
            "ytick.color": ink,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(18 / 2.54, 5.25))
    ax = fig.add_axes([0.095, 0.575, 0.88, 0.29])
    lower = fig.add_axes([0.095, 0.115, 0.88, 0.255])
    fig.text(0.095, 0.98, "(a) Discharge coverage, 1987–2025", va="top", fontsize=10)
    fig.text(
        0.095,
        0.465,
        "(b) Return levels and 95% bootstrap intervals",
        va="top",
        fontsize=10,
    )
    inside = hourly.year.isin(expanded)
    measured = hourly.observed_discharge_m3_s
    accepted_source = (
        hourly.source_quality_control.eq(3)
        & hourly.source_correction.isin([0, 1])
        & hourly.source_discharge_m3_s.ge(0)
    )
    excluded = hourly.source_discharge_m3_s.where(accepted_source & ~inside)
    ax.plot(hourly.index, excluded, color=gray, lw=0.55, zorder=1, rasterized=True)
    for mask, color in [
        (hourly.year.isin(reference), before_color),
        (hourly.year.isin(expanded - reference), after_color),
    ]:
        ax.plot(
            hourly.index,
            measured.where(mask),
            color=color,
            lw=0.6,
            zorder=2,
            rasterized=True,
        )
    fill = inside & measured.isna() & hourly.pot_discharge_m3_s.notna()
    anchor = (
        inside
        & measured.notna()
        & (fill.shift(1, fill_value=False) | fill.shift(-1, fill_value=False))
    )
    fill_values = hourly.pot_discharge_m3_s.where(fill | anchor)
    ax.plot(
        hourly.index,
        fill_values,
        color=filled_color,
        lw=0.85,
        zorder=3,
        rasterized=True,
    )
    ax.axhline(u, color="#626b72", lw=0.6, ls=(0, (3, 3)), zorder=0)
    ax.text(
        1.0,
        1.025,
        "Threshold $u=12.84$",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        color="#535e66",
        fontsize=8,
    )
    ax.set(
        xlim=(
            pd.Timestamp("1987-01-01", tz="UTC"),
            pd.Timestamp("2026-01-01", tz="UTC"),
        ),
        ylim=(0, 44.5),
        ylabel="Discharge ($\\mathrm{m^3\\,s^{-1}}$)",
        xlabel="Year",
    )
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.set_yticks([0, 10, 20, 30, 40])
    handles = [
        Line2D([], [], color=color, lw=1.1, label=label)
        for color, label in (
            (before_color, "Measured in reference years (15)"),
            (after_color, "Measured in additional years (13)"),
            (filled_color, "Reconstructed discharge"),
            (gray, "Measured in excluded years"),
        )
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.095, 0.936),
        ncol=2,
        frameon=False,
        borderaxespad=0,
        columnspacing=3.8,
        handletextpad=0.6,
        handlelength=2.4,
        labelspacing=0.65,
    )
    positions = {20: 2.0, 200: 1.0, 1000: 0.0}
    for sample, offset, color, marker, label in (
        ("before", 0.11, before_color, "o", "Reference (15 years)"),
        ("after", -0.11, after_color, "s", "Expanded (28 years)"),
    ):
        data = levels.loc[sample].loc[[20, 200, 1000]]
        point = data.return_level_m3_s.to_numpy()
        error = np.vstack((point - data.lower_95_m3_s, data.upper_95_m3_s - point))
        y = np.array([positions[p] for p in data.index]) + offset
        lower.errorbar(
            point,
            y,
            xerr=error,
            color=color,
            fmt=marker,
            ms=4,
            lw=1.1,
            capsize=3,
            capthick=0.9,
            markeredgewidth=0.8,
            markerfacecolor="white",
            label=label,
        )
    lower.set(
        xlim=(20, 70),
        ylim=(-0.48, 2.48),
        yticks=[2, 1, 0],
        yticklabels=["20", "200", "1000"],
        ylabel="Return period (years)",
        xlabel="Return level ($\\mathrm{m^3\\,s^{-1}}$)",
    )
    lower.set_xticks([20, 30, 40, 50, 60, 70])
    lower.legend(
        loc="upper left",
        bbox_to_anchor=(0, 1.25),
        frameon=False,
        ncol=2,
        borderaxespad=0,
        handletextpad=0.7,
        columnspacing=2.6,
    )
    for axes in (ax, lower):
        axes.set_axisbelow(True)
        axes.grid(axis="y" if axes is ax else "x", color="#e4e8ec", lw=0.55)
        axes.tick_params(direction="out", length=3, width=0.6, pad=4)
        axes.tick_params(which="minor", length=1.7, width=0.5)
        for spine in axes.spines.values():
            spine.set_color("#89939b")
    for extension in ("pdf", "png"):
        fig.savefig(
            figures / f"flood_frequency.{extension}", dpi=400, facecolor="white"
        )
    plt.close(fig)


def project(lon, lat):
    return (
        radius * np.radians(lon),
        radius * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)),
    )


def distance(row, origin):
    p1, p2 = np.radians([row.latitude, origin.latitude])
    dlon = np.radians(row.longitude - origin.longitude)
    h = np.sin((p1 - p2) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * earth_km * np.arcsin(np.sqrt(h))


def locator_lines(layer):
    points = pd.read_csv(map_folder / "locator.csv")
    points = points.loc[points.layer.eq(layer)]
    return [
        line[["x", "y"]].to_numpy() for _, line in points.groupby("line", sort=False)
    ]


def plot_stations():
    figures.mkdir(exist_ok=True)
    inventory = pd.read_csv(map_folder / "stations.csv", dtype={"station_id": str})
    ids = ["2.11.0", "2.265.0", "1083", "1223"]
    stations = inventory.set_index("station_id").loc[ids].reset_index()
    assert len(stations) == 4 and stations.station_id.is_unique
    origin = stations.loc[stations.station_id.eq("2.11.0")].iloc[0]
    stations["distance_km"] = [distance(row, origin) for row in stations.itertuples()]
    assert np.allclose(
        stations.loc[stations.station_id.ne("2.11.0"), "distance_km"],
        [50.52755, 45.39897, 49.70861],
        atol=2e-05,
    )
    for row in stations.itertuples():
        assert map_bounds[0] < row.longitude < map_bounds[1]
        assert map_bounds[2] < row.latitude < map_bounds[3]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 8,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelsize": 8,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "axes.linewidth": 0.65,
            "text.color": ink,
            "axes.labelcolor": ink,
            "xtick.color": ink,
            "ytick.color": ink,
            "pdf.fonttype": 42,
            "mathtext.fontset": "cm",
        }
    )
    extent = pd.read_csv(map_folder / "extent.csv").iloc[0]
    extent = extent[["xmin", "xmax", "ymin", "ymax"]].to_numpy(float)
    basemap_path = map_folder / "background.png"
    fig = plt.figure(figsize=(3.55, 3.75))
    ax = fig.add_axes([0.035, 0.09, 0.93, 0.88])
    ax.set_facecolor("#f6f6f1")
    ax.imshow(
        plt.imread(basemap_path), extent=extent, origin="upper", alpha=0.79, zorder=1
    )
    x0, y0 = project(origin.longitude, origin.latitude)
    offsets = {
        "2.265.0": (0, -13, "center"),
        "1083": (-6, 10, "right"),
        "1223": (-6, -12, "right"),
    }
    for row in stations.loc[stations.station_id.ne("2.11.0")].itertuples():
        x, y = project(row.longitude, row.latitude)
        (line,) = ax.plot([x0, x], [y0, y], color=donor_color, lw=1.25, zorder=4)
        line.set_path_effects(
            [
                patheffects.Stroke(linewidth=2.5, foreground="white"),
                patheffects.Normal(),
            ]
        )
        ax.scatter(x, y, s=30, color=donor_color, edgecolor="white", lw=0.8, zorder=6)
        fraction = {"2.265.0": 0.53, "1083": 0.59, "1223": 0.3}[row.station_id]
        angle = np.degrees(np.arctan2(y - y0, x - x0))
        if angle < -90 or angle > 90:
            angle += 180
        ax.text(
            x0 + fraction * (x - x0),
            y0 + fraction * (y - y0),
            f"{row.distance_km:.1f} km",
            rotation=angle,
            rotation_mode="anchor",
            ha="center",
            va="center",
            color=donor_color,
            fontsize=7.7,
            zorder=7,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.4, "alpha": 0.96},
        )
        dx, dy, align = offsets[row.station_id]
        ax.annotate(
            row.name,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=align,
            va="center",
            fontsize=8.2,
            linespacing=1.25,
            zorder=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.3},
        )
    ax.scatter(
        x0,
        y0,
        s=95,
        marker="*",
        color=target_color,
        edgecolor="white",
        lw=0.7,
        zorder=9,
    )
    ax.annotate(
        "Narsjø",
        (x0, y0),
        xytext=(7, -11),
        textcoords="offset points",
        color=target_color,
        fontsize=9.2,
        linespacing=1.25,
        zorder=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.2},
    )
    for label, lon, lat in [("Norway", 11.04, 62.56), ("Sweden", 12.49, 62.31)]:
        x, y = project(lon, lat)
        ax.text(
            x,
            y,
            label,
            fontsize=7.3,
            ha="center",
            color="#65717a",
            zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.2},
        )
    west, east, south, north = map_bounds
    ax.set_xlim(*project(np.array([west, east]), 0)[0])
    ax.set_ylim(*project(0, np.array([south, north]))[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set(xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_color("#7f8990")
    handles = [
        Line2D([], [], color=target_color, marker="*", ls="none", ms=8, label="Narsjø"),
        Line2D(
            [],
            [],
            color=donor_color,
            marker="o",
            ls="none",
            ms=4.5,
            label="Donor stations",
        ),
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        fontsize=7.5,
        handletextpad=0.4,
        borderpad=0.45,
        labelspacing=0.45,
        framealpha=0.95,
    )
    legend.get_frame().set_edgecolor("#c6ccd0")
    legend.set_zorder(12)
    bar_lon, bar_lat = (11.55, 61.895)
    angle = 2 * np.arcsin(np.sin(20 / (2 * earth_km)) / np.cos(np.radians(bar_lat)))
    bx1, by = project(bar_lon, bar_lat)
    bx2, _ = project(bar_lon + np.degrees(angle), bar_lat)
    ax.plot([bx1, bx2], [by, by], color=ink, lw=1.5, zorder=8)
    ax.plot([bx1, bx1], [by - 1800, by + 1800], color=ink, lw=0.8, zorder=8)
    ax.plot([bx2, bx2], [by - 1800, by + 1800], color=ink, lw=0.8, zorder=8)
    ax.annotate(
        "20 km",
        ((bx1 + bx2) / 2, by),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        fontsize=7.4,
        zorder=9,
    )
    ax.annotate(
        "",
        xy=(0.93, 0.91),
        xytext=(0.93, 0.84),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": ink, "lw": 0.8},
        zorder=9,
    )
    ax.text(
        0.93,
        0.94,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        zorder=9,
    )
    inset = ax.inset_axes([0.725, 0.035, 0.25, 0.33])
    inset_bounds = (3.0, 32.0, 54.0, 72.0)
    inset.set_facecolor("#e6eff3")
    for land in locator_lines("land"):
        inset.fill(land[:, 0], land[:, 1], color="#f4f3ec", ec="none", zorder=1)
    inset.add_collection(
        LineCollection(
            locator_lines("coast"), color="#85949c", linewidth=0.35, zorder=2
        )
    )
    inset.add_collection(
        LineCollection(
            locator_lines("borders"), color="#a2adb3", linewidth=0.3, zorder=3
        )
    )
    xx, yy = project(np.array([west, east]), np.array([south, north]))
    inset.add_patch(
        Rectangle(
            (xx[0], yy[0]),
            xx[1] - xx[0],
            yy[1] - yy[0],
            fill=False,
            ec=target_color,
            lw=1,
            zorder=5,
        )
    )
    ix, iy = project(np.array(inset_bounds[:2]), np.array(inset_bounds[2:]))
    inset.set(xlim=ix, ylim=iy, xticks=[], yticks=[])
    inset.set_aspect("equal", adjustable="box")
    for spine in inset.spines.values():
        spine.set_color("#8a969d")
        spine.set_linewidth(0.6)
    fig.text(
        0.035,
        0.025,
        "Esri and contributors · Inset: Natural Earth",
        ha="left",
        va="bottom",
        fontsize=6.0,
        color="#65717a",
    )
    for extension in ("pdf", "png"):
        fig.savefig(figures / f"station_map.{extension}", dpi=450, facecolor="white")
    plt.close(fig)
