
import math
import statistics

def _mean_last(series, window):
    """Mean over the last `window` points of a 1D list."""
    if not series:
        return None
    if window is None or window <= 0 or window > len(series):
        data = series
    else:
        data = series[-window:]
    return float(sum(data) / len(data))

def summarize_module(mod, eval_window=52):
    """
    Given a loaded scenario module (after simulation has run),
    compute a compact summary dict with key system-level metrics.
    """
    res = {}

    # Time index length inferred from flow_share or insurance
    T = None
    try:
        T = len(mod.flow_share_lvl["county"])
    except Exception:
        try:
            T = len(mod.insurance_spending["total"])
        except Exception:
            pass
    if T is None:
        return res

    w = eval_window if (eval_window is not None and eval_window > 0) else T
    w = min(w, T)

    idx_start = T - w

    # 1) Visit share: county vs primary (secondary+township)
    try:
        county_share = mod.flow_share_lvl["county"][idx_start:]
        primary_share = getattr(mod, "combined_primary_flow_share", None)
        if primary_share is None or len(primary_share) < T:
            # fallback: 1 - county
            primary_share = [1.0 - c for c in mod.flow_share_lvl["county"]]
        primary_share = primary_share[idx_start:]

        res["flow_share_county_mean"] = _mean_last(mod.flow_share_lvl["county"], w)
        res["flow_share_primary_mean"] = _mean_last(primary_share, w)
    except Exception:
        pass

    # 2) Insurance spending
    try:
        ins = mod.insurance_spending
        res["insurance_total_mean"] = _mean_last(ins["total"], w)
        # county & primary shares
        county_sh = ins.get("county_share", [])
        sec_sh = ins.get("secondary_share", [])
        town_sh = ins.get("township_share", [])
        if county_sh:
            res["insurance_county_share_mean"] = _mean_last(county_sh, w)
        if sec_sh and town_sh:
            primary_share_series = [
                (sec_sh[t] if t < len(sec_sh) else 0.0) +
                (town_sh[t] if t < len(town_sh) else 0.0)
                for t in range(len(county_sh))
            ]
            # align with window
            res["insurance_primary_share_mean"] = _mean_last(primary_share_series, w)
    except Exception:
        pass

    # 3) Referral success rate
    try:
        rs = mod.referral_success_rate
        res["referral_success_rate_mean"] = _mean_last(rs, w)
    except Exception:
        pass

    # 4) Severity: county vs primary (flow-weighted)
    try:
        sev_stats = mod.sev_stats_lvl
        sev_county = sev_stats["county"]["mean"]
        sev_secondary = sev_stats["secondary"]["mean"]
        sev_township = sev_stats["township"]["mean"]
        flow_sec = mod.flow_share_lvl["secondary"]
        flow_town = mod.flow_share_lvl["township"]

        # Simple means
        res["severity_county_mean"] = _mean_last(sev_county, w)

        # Flow-weighted primary mean per time step, then average
        primary_vals = []
        for t in range(idx_start, T):
            w_sec = flow_sec[t] if t < len(flow_sec) else 0.0
            w_town = flow_town[t] if t < len(flow_town) else 0.0
            denom = w_sec + w_town
            if denom <= 0:
                # simple average
                val = 0.5 * (sev_secondary[t] + sev_township[t])
            else:
                val = (w_sec * sev_secondary[t] + w_town * sev_township[t]) / denom
            primary_vals.append(val)
        if primary_vals:
            res["severity_primary_mean"] = float(sum(primary_vals) / len(primary_vals))
    except Exception:
        pass

    # 5) Capacity (bed-days) — last value + mean over window
    try:
        cap = mod.capacity_trends
        for level in ["county", "secondary", "township"]:
            series = cap.get(level, [])
            if series:
                res[f"capacity_{level}_last"] = float(series[-1])
                res[f"capacity_{level}_mean"] = _mean_last(series, w)
    except Exception:
        pass

    # Aggregate primary capacity (secondary + township)
    try:
        cap = mod.capacity_trends
        sec = cap.get("secondary", [])
        town = cap.get("township", [])
        if sec and town:
            primary_series = []
            for t in range(min(len(sec), len(town))):
                primary_series.append(sec[t] + town[t])
            if primary_series:
                res["capacity_primary_last"] = float(primary_series[-1])
                res["capacity_primary_mean"] = _mean_last(primary_series, w)
    except Exception:
        pass

    return res
