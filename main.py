#!/usr/bin/env python3
"""Forecast a launch day's Twitch viewers, hours 2..24, as five quantiles.

The program reads every ``*.in`` case file in the input directory and writes
one ``<id>.out`` forecast per case.  It is a launch-shape forecaster: the
hour-1 print anchors the scale, and the shape of the rest of the day is
learned at run time from the case's own comparables (past big launches).

Model, per forecast hour h (2..24):
  * For every comparable launch, take its release-day hourly counts and form
    log-ratios ``ln(viewers_h / viewers_1)`` against its own hour-1 print.
  * Fit a local (kernel-weighted) linear regression of the log-ratio on
    ``ln(viewers_1)`` across the comparables, with a Gaussian kernel of
    bandwidth 1.5 in log space, evaluated at this case's hour-1 print.
    Big launches grow less over the day than small ones (the fitted slope
    runs from about 0 in the small hours to about -0.9 in the evening), so
    conditioning on the hour-1 size matters more than any single average
    curve.  The weighted residual spread of the fit is the log-spread.
  * Add a rolling-midnight unlock bump.  These launches unlock at local
    midnight, so the US East coast joins at midnight Eastern (04:00Z in
    daylight time, 05:00Z otherwise) and the US West coast three hours
    later; the comparable pool mixes unlock structures, which smooths that
    step away, so the median is lifted at the unlock hours.  The bump
    sizes were calibrated on the four practice launches.
  * Quantiles are ``v1 * exp(mu + bump + z_tau * sigma * 0.7)`` with the
    standard normal quantiles at the five levels.

Everything is deterministic: fixed iteration order, no randomness, no clock
or network use.  Standard library only.

Fallbacks keep the program total: when a forecast hour has too few
comparable points the fit falls back to a weighted median of the available
ratios (or a flat ratio of 1.0) with the default spread, so sparse data
can never crash a run or leave an hour unforecast.
"""

import json
import math
import os
from datetime import date, datetime, timedelta

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

BANDWIDTH = 1.5        # Gaussian kernel width in log(viewers_1)
SPREAD_K = 0.7         # shrink on the residual log-spread
MIN_POINTS = 3         # fewest comparables for a local fit
DEFAULT_SIGMA = 0.8    # fallback log-spread when a fit is impossible
SLOPE_LO = -1.2        # clamp on the fitted slope, for degenerate fits
SLOPE_HI = 0.5
SIGMA_LO = 0.15
SIGMA_HI = 1.5

# Log-bump at the US unlock hours (practice-calibrated).  The East coast
# unlocks first; the three entries are the unlock hour and the two after
# it.  The West coast unlocks three hours later; two entries.
EAST_BUMP = (0.50, 0.48, 0.38)
WEST_BUMP = (0.18, 0.16)

SELF_NAME = "Grand Theft Auto VI"   # the pre-release category comparable


def parse_time(text):
    """Parse an ISO-8601 timestamp into an aware datetime."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def is_us_daylight(day):
    """True when `day` falls in US daylight time (2nd Sun Mar - 1st Sun Nov)."""
    def nth_sunday(year, month, n):
        first = date(year, month, 1)
        return first + timedelta(days=(6 - first.weekday()) % 7 + 7 * (n - 1))
    return nth_sunday(day.year, 3, 2) <= day < nth_sunday(day.year, 11, 1)


def weighted_median(pairs):
    """Weighted median of (value, weight) pairs."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= 0.5 * total:
            return value
    return ordered[-1][0]


def local_fit(points, x0):
    """Kernel-weighted local linear fit of y on x, predicted at x0.

    `points` is a list of (x, y).  Returns (mu, sigma): the fitted value at
    x0 and the weighted residual standard deviation around the fit.  Falls
    back to a weighted median and the default spread when there are too few
    points or the fit degenerates.
    """
    weights = [math.exp(-0.5 * ((x - x0) / BANDWIDTH) ** 2)
               for x, _ in points]
    total = sum(weights)
    if len(points) < MIN_POINTS or total <= 0.0:
        mu = weighted_median([(y, w) for (_, y), w in zip(points, weights)]) \
            if points else 0.0
        return mu, DEFAULT_SIGMA
    mx = sum(w * x for w, (x, _) in zip(weights, points)) / total
    my = sum(w * y for w, (_, y) in zip(weights, points)) / total
    sxx = sum(w * (x - mx) ** 2 for w, (x, _) in zip(weights, points))
    if sxx <= 1e-12:
        return my, DEFAULT_SIGMA
    slope = sum(w * (x - mx) * (y - my)
                for w, (x, y) in zip(weights, points)) / sxx
    slope = max(SLOPE_LO, min(SLOPE_HI, slope))
    intercept = my - slope * mx
    mu = intercept + slope * x0
    var = sum(w * (y - (intercept + slope * x)) ** 2
              for w, (x, y) in zip(weights, points)) / total
    sigma = math.sqrt(var) if var > 0.0 else DEFAULT_SIGMA
    return mu, max(SIGMA_LO, min(SIGMA_HI, sigma))


def release_day_curve(comp):
    """A comparable's release-day hourly viewers as {hour offset: viewers}.

    Returns None when the comparable has no usable release day (missing
    hour 0, or fewer than 20 of the 24 hours present).  The target game's
    own pre-release category history is not a launch and is skipped.
    """
    if SELF_NAME in comp.get("name", ""):
        return None
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    hours = {}
    for point in comp.get("hourly", []):
        try:
            at = parse_time(point["from"])
            viewers = point["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        offset = (at - day_start).total_seconds() / 3600.0
        if 0 <= offset < 24 and viewers and viewers > 0:
            hours[int(offset)] = viewers
    if 0 not in hours or len(hours) < 20:
        return None
    return hours


def forecast_case(case):
    """Build the {"hours": [...]} forecast object for one case."""
    known = case.get("known") or []
    first = known[0].get("viewers") if known else None
    v1 = first if first and first > 0 else 1
    log_v1 = math.log(v1)

    # Per-hour regression points: (log viewers_1, log ratio) per comparable.
    by_hour = {}
    for comp in case.get("comparables", []):
        curve = release_day_curve(comp)
        if curve is None:
            continue
        base = math.log(curve[0])
        for hour in range(1, 24):
            if hour in curve:
                by_hour.setdefault(hour, []).append(
                    (base, math.log(curve[hour] / curve[0])))

    # Rolling-midnight unlock hours (UTC) for this release date.
    day = parse_time(case["day_start"]).date()
    daylight = is_us_daylight(day)
    east_first = (4 if daylight else 5) + 1   # first East-unlocked hour
    west_first = (7 if daylight else 8) + 1   # first West-unlocked hour

    rows = []
    for row in sorted(case.get("forecast", []), key=lambda r: r["hour"]):
        hour = row["hour"]
        mu, sigma = local_fit(by_hour.get(hour - 1, []), log_v1)
        bump = 0.0
        for i, size in enumerate(EAST_BUMP):
            if hour == east_first + i:
                bump += size
        for i, size in enumerate(WEST_BUMP):
            if hour == west_first + i:
                bump += size
        spread = sigma * SPREAD_K
        quantiles = []
        floor = 0.0
        for z in Z:
            value = v1 * math.exp(mu + bump + z * spread)
            value = round(value, 2)
            value = max(0.01, min(1e9, value))
            value = max(value, floor)   # keep the row non-decreasing
            floor = value
            quantiles.append(value)
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input/")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output/")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(name for name in os.listdir(in_dir)
                   if name.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        forecast = forecast_case(case)
        stem = name[:-3]
        with open(os.path.join(out_dir, stem + ".out"), "w",
                  encoding="utf-8") as fh:
            json.dump(forecast, fh)


if __name__ == "__main__":
    main()
