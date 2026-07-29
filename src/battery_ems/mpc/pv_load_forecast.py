"""
Simple deterministic PV/load forecast for the demo's joint cooling+PV+battery
MPC -- NOT a port of the private repo's live Chronos-2 + 14-day-history
pipeline (chronos_pv_load_provider.py + data_processing/aggregator.py),
which needs a whole real-meter preprocessing chain out of scope for this
demo (see README's "what's simplified" section). A smooth clear-sky/diurnal
projection instead, using the same functional shape
emulators/synthetic_building/plant.py's ground-truth `_diurnal_pv_load` does
(minus its noise term -- a forecast is a smooth prediction, not a sample).
"""
import math
from datetime import datetime, timedelta

STEP_MINUTES = 5
PV_CAPACITY_KW = 5.0


class PVLoadForecastProvider:
    def get(self, horizon_steps: int, now: datetime) -> tuple[list[float], list[float]]:
        pv, load = [], []
        for i in range(horizon_steps):
            t = now + timedelta(minutes=i * STEP_MINUTES)
            hour = t.hour + t.minute / 60.0
            if 6 <= hour <= 19:
                solar_wm2 = max(0.0, 700.0 * math.sin(math.pi * (hour - 6) / 13))
            else:
                solar_wm2 = 0.0
            pv.append(max(0.0, PV_CAPACITY_KW * (solar_wm2 / 700.0)))
            evening_bump = 0.6 * math.exp(-0.5 * ((hour - 19.5) / 2.0) ** 2)
            load.append(max(0.05, 0.35 + evening_bump))
        return pv, load
