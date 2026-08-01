"""
Simple deterministic PV/load forecast for the demo's joint cooling+PV+battery
MPC. In the real deplyoment, Chronos2 is applied for forecasting.
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
