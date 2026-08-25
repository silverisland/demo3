"""Station-level monotonic time registration for 15-minute PV curves."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


POINT_PER_HOUR = 4
POINTS_PER_DAY = 24 * POINT_PER_HOUR

MIN_VALID_SLOTS_PER_DAY = 72
MIN_SELECTED_DAYS = 5
USE_HIGH_ENERGY_DAYS = True
HIGH_ENERGY_DAY_FRACTION = 0.40

N_WARP_KNOTS = 5
MAX_TIME_SHIFT_HOURS = 1.0
MIN_LOCAL_SLOPE = 0.75
MAX_LOCAL_SLOPE = 1.33
IDENTITY_PENALTY = 0.10
SMOOTHNESS_PENALTY = 0.05
GRADIENT_LOSS_WEIGHT = 0.20
DAYLIGHT_THRESHOLD = 0.02
N_TEMPLATE_ITERATIONS = 2
MIN_ALIGNMENT_IMPROVEMENT = 0.01

CURVE_GRID = np.arange(POINTS_PER_DAY, dtype=np.float64) / POINTS_PER_DAY
CANONICAL_KNOTS = np.linspace(0.0, 1.0, N_WARP_KNOTS)
MINUTES_PER_DAY = 24.0 * 60.0
MAX_SHIFT_FRACTION = MAX_TIME_SHIFT_HOURS / 24.0


def _last_value(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0 or not np.isfinite(values[-1]):
        return np.nan
    return float(values[-1])


def _median_curve(
    df,
    capacity,
    mapping_start,
    mapping_end,
    timestamp_col,
    power_history_col,
    history_last_offset_minutes,
):
    data = df[
        df[timestamp_col].between(mapping_start, mapping_end)
    ].copy()
    if data.empty:
        raise ValueError("No data in the curve-registration period")

    timestamp = (
        pd.to_datetime(data[timestamp_col])
        + pd.to_timedelta(history_last_offset_minutes, unit="m")
    )
    power = data[power_history_col].map(_last_value) / capacity
    samples = pd.DataFrame(
        {
            "date": timestamp.dt.date,
            "slot": (
                timestamp.dt.hour * POINT_PER_HOUR
                + timestamp.dt.minute // 15
            ),
            "power": power,
        }
    ).dropna()
    daily = samples.pivot_table(
        index="date",
        columns="slot",
        values="power",
        aggfunc="median",
    ).reindex(columns=np.arange(POINTS_PER_DAY))
    daily = daily[daily.notna().sum(axis=1) >= MIN_VALID_SLOTS_PER_DAY]
    daily = daily.interpolate(axis=1, limit_direction="both").dropna()
    if len(daily) < MIN_SELECTED_DAYS:
        raise ValueError(
            f"Only {len(daily)} valid days in the curve-registration period"
        )

    if USE_HIGH_ENERGY_DAYS:
        energy = daily.clip(lower=0.0).sum(axis=1)
        n_selected = max(
            MIN_SELECTED_DAYS,
            int(np.ceil(len(daily) * HIGH_ENERGY_DAY_FRACTION)),
        )
        selected = energy.nlargest(min(n_selected, len(daily))).index
        daily = daily.loc[selected]

    return daily.median(axis=0).to_numpy(dtype=np.float64)


def _normalize_shape(curve):
    """Remove stable amplitude only while estimating the time mapping."""
    curve = np.clip(np.asarray(curve, dtype=np.float64), 0.0, None)
    daylight = curve > DAYLIGHT_THRESHOLD
    if daylight.sum() < 8:
        raise ValueError("Too few daylight points for shape normalization")
    scale = float(np.quantile(curve[daylight], 0.95))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError(f"Invalid P95 shape scale: {scale}")
    return curve / scale, scale


def apply_daily_warp(curve, source_knots):
    """Return q(tau)=p(psi(tau)) and t=psi(tau)."""
    source_position = np.interp(
        CURVE_GRID,
        CANONICAL_KNOTS,
        source_knots,
    )
    registered = np.interp(
        source_position,
        np.r_[CURVE_GRID, 1.0],
        np.r_[curve, curve[0]],
    )
    return registered, source_position


def inverse_daily_warp(registered_curve, source_position):
    """Approximately restore a registered 96-point curve."""
    if not np.all(np.diff(source_position) > 0):
        raise ValueError("The time mapping must be strictly increasing")

    canonical_position = np.interp(
        CURVE_GRID,
        np.r_[source_position, 1.0],
        np.r_[CURVE_GRID, 1.0],
    )
    return np.interp(
        canonical_position,
        np.r_[CURVE_GRID, 1.0],
        np.r_[registered_curve, registered_curve[0]],
    )


def _fit_warp(curve, template):
    def unpack(x):
        return np.r_[0.0, x, 1.0]

    def objective(x):
        source_knots = unpack(x)
        registered, _ = apply_daily_warp(curve, source_knots)
        daylight = (
            (template > DAYLIGHT_THRESHOLD)
            | (registered > DAYLIGHT_THRESHOLD)
        )
        if daylight.sum() < 8:
            daylight = np.ones_like(template, dtype=bool)

        fit_loss = np.mean(
            (registered[daylight] - template[daylight]) ** 2
        )
        gradient_loss = np.mean(
            (
                np.gradient(registered)[daylight]
                - np.gradient(template)[daylight]
            )
            ** 2
        )
        identity_loss = np.mean(
            (source_knots - CANONICAL_KNOTS) ** 2
        )
        smoothness_loss = np.mean(
            np.diff(source_knots, n=2) ** 2
        )
        return (
            fit_loss
            + GRADIENT_LOSS_WEIGHT * gradient_loss
            + IDENTITY_PENALTY * identity_loss
            + SMOOTHNESS_PENALTY * smoothness_loss
        )

    def local_slopes(x):
        return np.diff(unpack(x)) / np.diff(CANONICAL_KNOTS)

    result = minimize(
        objective,
        CANONICAL_KNOTS[1:-1],
        method="SLSQP",
        bounds=[
            (
                max(0.0, knot - MAX_SHIFT_FRACTION),
                min(1.0, knot + MAX_SHIFT_FRACTION),
            )
            for knot in CANONICAL_KNOTS[1:-1]
        ],
        constraints=[
            {
                "type": "ineq",
                "fun": lambda x: local_slopes(x) - MIN_LOCAL_SLOPE,
            },
            {
                "type": "ineq",
                "fun": lambda x: MAX_LOCAL_SLOPE - local_slopes(x),
            },
        ],
        options={"maxiter": 800, "ftol": 1e-11, "disp": False},
    )
    source_knots = (
        unpack(result.x)
        if result.success
        else CANONICAL_KNOTS.copy()
    )
    registered, source_position = apply_daily_warp(curve, source_knots)
    before = _alignment_rmse(curve, template)
    after = _alignment_rmse(registered, template)
    improvement = (before - after) / before if before > 0 else 0.0
    accepted = bool(
        result.success
        and improvement >= MIN_ALIGNMENT_IMPROVEMENT
    )
    if not accepted:
        source_knots = CANONICAL_KNOTS.copy()
        registered, source_position = apply_daily_warp(
            curve,
            source_knots,
        )
        after = _alignment_rmse(registered, template)
        improvement = (before - after) / before if before > 0 else 0.0

    return {
        "registered": registered,
        "position": source_position,
        "knots": source_knots,
        "before": before,
        "after": after,
        "improvement": improvement,
        "accepted": accepted,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }


def _alignment_rmse(curve, template):
    daylight = (
        (curve > DAYLIGHT_THRESHOLD)
        | (template > DAYLIGHT_THRESHOLD)
    )
    if daylight.sum() < 8:
        daylight = np.ones_like(template, dtype=bool)
    return float(
        np.sqrt(np.mean((curve[daylight] - template[daylight]) ** 2))
    )


def fit_station_warps(
    station_frames,
    source_stations,
    target_station,
    station_capacity,
    source_mapping_start,
    source_mapping_end,
    target_mapping_start,
    target_mapping_end,
    timestamp_col="timestamp_win",
    power_history_col="observe_power",
    history_last_offset_minutes=0,
):
    """
    Fit a source-only common template, then fit every source and target station
    to that fixed template.
    """
    stations = list(source_stations) + [target_station]
    capacity_curves = {}
    shape_curves = {}
    shape_scales = {}
    for station in stations:
        mapping_start, mapping_end = (
            (target_mapping_start, target_mapping_end)
            if station == target_station
            else (source_mapping_start, source_mapping_end)
        )
        capacity_curves[station] = _median_curve(
            station_frames[station],
            float(station_capacity[station]),
            mapping_start,
            mapping_end,
            timestamp_col,
            power_history_col,
            history_last_offset_minutes,
        )
        shape_curves[station], shape_scales[station] = _normalize_shape(
            capacity_curves[station]
        )

    template = np.median(
        np.stack([shape_curves[station] for station in source_stations]),
        axis=0,
    )
    for i in range(N_TEMPLATE_ITERATIONS):
        registered = [
            _fit_warp(shape_curves[station], template)["registered"]
            for station in source_stations
        ]
        new_template = np.median(np.stack(registered), axis=0)
        change = np.sqrt(np.mean((new_template - template) ** 2))
        print(
            f"template iteration {i + 1}/{N_TEMPLATE_ITERATIONS}: "
            f"RMSE change={change:.6f}"
        )
        template = new_template

    station_warps = {}
    print("\n[Curve registration]")
    for station in stations:
        fit = _fit_warp(shape_curves[station], template)
        registered_capacity, position = apply_daily_warp(
            capacity_curves[station],
            fit["knots"],
        )
        restored = inverse_daily_warp(registered_capacity, position)
        roundtrip = np.sqrt(
            np.mean((restored - capacity_curves[station]) ** 2)
        )
        slopes = np.diff(fit["knots"]) / np.diff(CANONICAL_KNOTS)
        max_shift = np.max(
            np.abs(fit["knots"] - CANONICAL_KNOTS)
        ) * MINUTES_PER_DAY
        print(
            f"station={station:<20} accepted={str(fit['accepted']):<5} "
            f"P95={shape_scales[station]:.4f} "
            f"before={fit['before']:.6f} after={fit['after']:.6f} "
            f"improve={100.0 * fit['improvement']:.2f}% "
            f"shift={max_shift:.1f}min "
            f"slope=[{slopes.min():.3f}, {slopes.max():.3f}] "
            f"roundtrip={roundtrip:.8f}"
        )
        station_warps[station] = fit["knots"]

    return station_warps


def _absolute_minutes(timestamp):
    timestamp = pd.Timestamp(timestamp)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.value / (60.0 * 1e9)


def _split_day(absolute_minutes):
    absolute_minutes = np.asarray(absolute_minutes, dtype=np.float64)
    day_start = (
        np.floor(absolute_minutes / MINUTES_PER_DAY)
        * MINUTES_PER_DAY
    )
    fraction = np.clip(
        (absolute_minutes - day_start) / MINUTES_PER_DAY,
        0.0,
        1.0,
    )
    return day_start, fraction


def _physical_to_canonical(physical_minutes, source_knots):
    day_start, physical_fraction = _split_day(physical_minutes)
    canonical_fraction = np.interp(
        physical_fraction,
        source_knots,
        CANONICAL_KNOTS,
    )
    return day_start + canonical_fraction * MINUTES_PER_DAY


def _canonical_to_physical(canonical_minutes, source_knots):
    day_start, canonical_fraction = _split_day(canonical_minutes)
    physical_fraction = np.interp(
        canonical_fraction,
        CANONICAL_KNOTS,
        source_knots,
    )
    return day_start + physical_fraction * MINUTES_PER_DAY


def register_history(
    observe_power,
    timestamp_win,
    capacity,
    source_knots,
    input_len=96,
    history_last_offset_minutes=0,
):
    """Register one historical array to an input_len-point canonical window."""
    history = np.asarray(observe_power, dtype=np.float64).reshape(-1)
    if len(history) < input_len:
        raise ValueError(
            f"History length {len(history)} is smaller than {input_len}"
        )
    history = history / float(capacity)

    history_end = (
        _absolute_minutes(timestamp_win)
        + history_last_offset_minutes
    )
    physical_history_time = (
        history_end
        - np.arange(len(history) - 1, -1, -1) * 15.0
    )

    canonical_end = float(
        _physical_to_canonical(history_end, source_knots)
    )
    canonical_input_time = (
        canonical_end
        - np.arange(input_len - 1, -1, -1) * 15.0
    )
    required_physical_time = _canonical_to_physical(
        canonical_input_time,
        source_knots,
    )

    return np.interp(
        required_physical_time,
        physical_history_time,
        history,
    ).astype(np.float32)


def physical_to_canonical_hour(timestamp, source_knots):
    """Map a physical target timestamp to the common time-of-day coordinate."""
    canonical_minutes = float(
        _physical_to_canonical(
            _absolute_minutes(timestamp),
            source_knots,
        )
    )
    canonical_timestamp = pd.to_datetime(canonical_minutes, unit="m")
    return (
        canonical_timestamp.hour
        + canonical_timestamp.minute / 60.0
        + canonical_timestamp.second / 3600.0
    )
