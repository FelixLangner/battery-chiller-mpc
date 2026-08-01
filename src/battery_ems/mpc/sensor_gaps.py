import logging
import math

import numpy as np

from battery_ems.mpc.fallback_state import FallbackState
from battery_ems.mpc.rc_observer import RCObserver

log = logging.getLogger(__name__)


def _fill_from_sibling(measurements: dict, key: str, source_key: str) -> None:
    """The two 'N-minutes-ago' fields have no fallback source of their own --
    if missing, they're just the already-resolved current reading held over."""
    if math.isnan(measurements[key]):
        measurements[key] = measurements[source_key]
        log.warning(f"{key} NaN — using current {source_key}")


def fill_sensor_gaps(measurements: dict, observer: RCObserver, rc_models: dict,
                      battery_capacity_kwh: float, fallback: FallbackState) -> dict:
    """
    Replace NaN sensor readings with the MPC's own prediction from the last
    successful solve where one exists, falling back further to
    observer/last-known values only if no prior solve is available yet.
    """
    state = observer._state
    predicted_list = fallback.predicted_next
    idx = fallback.consecutive_failures
    predicted = predicted_list[idx] if predicted_list is not None and idx < len(predicted_list) else None

    if math.isnan(measurements["T_amb"]):
        if predicted is not None:
            measurements["T_amb"] = predicted["T_amb"]
            log.warning(f"T_amb NaN — using previous solve's forecast: {measurements['T_amb']:.1f}°C")
        else:
            measurements["T_amb"] = 20.0
            log.warning(f"T_amb NaN — no prior solve yet, flat fallback: {measurements['T_amb']:.1f}°C")

    _fill_from_sibling(measurements, "T_amb_5min_ago", "T_amb")

    if math.isnan(measurements["T_sup"]):
        if predicted is not None:
            measurements["T_sup"] = predicted["T_sup"]
            log.warning(f"T_sup NaN — using previous solve's prediction: {measurements['T_sup']:.1f}°C")
        else:
            measurements["T_sup"] = state.get("T_sup_last", 16.0)
            log.warning(f"T_sup NaN — no prior solve yet, last-known fallback: {measurements['T_sup']:.1f}°C")

    _fill_from_sibling(measurements, "T_sup_5min_ago", "T_sup")

    for r, model in rc_models.items():
        if math.isnan(measurements["room_temps"].get(r, float("nan"))):
            C = np.atleast_2d(np.array(model["C"]))
            x = np.array(observer.x_state[r])
            measurements["room_temps"][r] = float((C @ x).item())
            log.warning(f"{r} room temp NaN — model prediction: {measurements['room_temps'][r]:.1f}°C")

        history = measurements["room_temps_history"].get(r, [])
        n_nan = sum(1 for v in history if math.isnan(v))
        if n_nan:
            log.warning(
                f"{r} room_temps_history has {n_nan}/{len(history)} NaN sub-step(s) this cycle "
                f"— observer will run open-loop (no Kalman correction) for those."
            )

        if math.isnan(measurements["Q_fan_measured"].get(r, float("nan"))):
            if predicted is not None:
                measurements["Q_fan_measured"][r] = predicted["Q_fan"][r]
                log.warning(f"{r} Q_fan NaN — using previous solve's prediction: "
                            f"{measurements['Q_fan_measured'][r]:.3f} kW")
            else:
                measurements["Q_fan_measured"][r] = 0.0
                log.warning(f"{r} Q_fan NaN — no prior solve yet, substituting 0.0 kW")

    if math.isnan(measurements["chiller_running"]):
        if fallback.last_plan is not None:
            measurements["chiller_running"] = float(fallback.last_plan["Chiller_Command"])
            log.warning(
                f"chiller_running NaN — falling back to last commanded state: "
                f"{'ON' if measurements['chiller_running'] else 'OFF'}"
            )
        else:
            measurements["chiller_running"] = 0.0
            log.warning("chiller_running NaN — no prior solve yet, conservative fallback: OFF")

    if math.isnan(measurements["battery_soc_kwh"]):
        if predicted is not None and "SOC" in predicted:
            measurements["battery_soc_kwh"] = predicted["SOC"]
            log.warning(f"battery_soc_kwh NaN — using previous solve's prediction: "
                        f"{measurements['battery_soc_kwh']:.2f} kWh")
        else:
            measurements["battery_soc_kwh"] = battery_capacity_kwh * 0.5
            log.warning(f"battery_soc_kwh NaN — no prior solve yet, 50% fallback: "
                        f"{measurements['battery_soc_kwh']:.2f} kWh")

    return measurements
