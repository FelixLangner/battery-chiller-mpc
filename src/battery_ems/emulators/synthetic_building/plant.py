"""
SyntheticBuilding: the "digital twin" ground-truth building. Owns 5 rooms'
RCPlant instances, a chiller with its own T_sup response (mirroring the
ON/OFF regime structure `gurobipy_mpc.py` expects, but as a SEPARATE set of
hand-authored constants from the MPC's own belief in
controllers/mpc_plant_power_coefs.json -- a fitted model is never a perfect
match to the real system it controls, and keeping these independent is more
realistic than trivially matching them by construction), and simple diurnal
weather/PV-irradiance generation.

Advances one 5-min physics step at a time via step(), writing the resulting
readings into a shared TimeSeriesStore -- the same one the synthetic
`EnergyDataInterface` (interfaces/influx.py) reads from, so `run_mpc.py`'s
control loop and this simulator interact only through that store plus
`PLC_API` (interfaces/PLC_API.py), exactly mirroring how the private repo's
`run_mpc.py` interacts with the real building only through InfluxDB reads
and PLC writes.
"""
import math
from datetime import datetime, timedelta

import numpy as np

from battery_ems.emulators.synthetic_building.rc_plant import RCPlant, ROOM_PARAMS

DT_SECONDS = 300

# Ground-truth chiller physics (independent of the MPC's own fitted belief
# in controllers/mpc_plant_power_coefs.json -- see module docstring).
CHILLER_ON_T_SUP_TARGET = 8.0
OFF_INTERCEPT = 0.85
OFF_T_SUP_CURRENT = 0.90
OFF_ENVTMP = 0.032
OFF_TOTAL_LOAD = -0.28

# Ground-truth per-room fan cooling capacity (kW/°C), independent of
# controllers/mpc_unified_fan_coefficients.json for the same reason.
FAN_SLOPE_KW_PER_C = {
    "room_1": -0.17,
    "room_2": -0.19,
    "room_3": -0.21,
    "room_4": -0.27,
    "room_5": -0.30,
}

# Supply-setpoint threshold below which the (hardware) chiller thermostat is
# considered "commanded on" -- matches control_writer.py's two setpoints
# (CHILLER_ON_SETPOINT=8, CHILLER_OFF_SETPOINT=20), midpoint split.
CHILLER_COMMAND_THRESHOLD_C = 14.0


def _diurnal_weather(t: datetime, rng: np.random.Generator) -> tuple[float, float]:
    """Simple plausible diurnal ambient temp (°C) + solar irradiance (W/m²)."""
    hour = t.hour + t.minute / 60.0
    T_amb = 24.0 + 6.0 * math.sin(2 * math.pi * (hour - 8) / 24) + rng.normal(0, 0.3)
    if 6 <= hour <= 19:
        solar = max(0.0, 700.0 * math.sin(math.pi * (hour - 6) / 13)) + rng.normal(0, 15)
        solar = max(0.0, solar)
    else:
        solar = 0.0
    return T_amb, solar


class SyntheticBuilding:
    def __init__(self, store, start_time: datetime, seed: int = 42):
        self.store = store
        self.time = start_time
        self.rng = np.random.default_rng(seed)
        self.rooms = {r: RCPlant(p, x0=(23.0, 23.0)) for r, p in ROOM_PARAMS.items()}
        self.T_sup = 18.0
        self._commanded_setpoint = 20.0  # starts "off"
        self.fan_speed: dict[str, str] = {r: "off" for r in self.rooms}

    # -- PLC-facing interface (called by the synthetic PLC_API) -----------

    def set_supply_setpoint(self, setpoint: float) -> None:
        self._commanded_setpoint = setpoint

    def set_fan_speed(self, room: str, speed: str) -> None:
        self.fan_speed[room] = speed

    def reset_all_fans(self, speed: str) -> None:
        for r in self.rooms:
            self.fan_speed[r] = speed

    # -- Physics ------------------------------------------------------------

    @property
    def chiller_on(self) -> bool:
        return self._commanded_setpoint <= CHILLER_COMMAND_THRESHOLD_C

    def step(self) -> dict:
        self.time = self.time + timedelta(seconds=DT_SECONDS)
        T_amb, solar_wm2 = _diurnal_weather(self.time, self.rng)
        solar_kwm2 = solar_wm2 / 1000.0

        room_Q_kw = {}
        for r, plant in self.rooms.items():
            is_on = self.fan_speed[r] == "normal"
            if is_on:
                Q = FAN_SLOPE_KW_PER_C[r] * (plant.T_room - self.T_sup)
                Q = min(Q, 0.0)  # cooling-only: never heats the room
            else:
                Q = 0.0
            room_Q_kw[r] = Q

        Total_Q = sum(room_Q_kw.values())
        if self.chiller_on:
            self.T_sup = CHILLER_ON_T_SUP_TARGET
        else:
            self.T_sup = (OFF_INTERCEPT + OFF_T_SUP_CURRENT * self.T_sup +
                          OFF_ENVTMP * T_amb + OFF_TOTAL_LOAD * Total_Q)
        self.T_sup = max(2.0, min(45.0, self.T_sup))

        for r, plant in self.rooms.items():
            plant.step(T_amb=T_amb, Q_fan_kw=room_Q_kw[r], Solar_kw_m2=solar_kwm2)

        readings = {
            "EnvTmp": T_amb,
            "GlobIrradHoriz": solar_wm2,
            "T_sup": self.T_sup,
            "chiller_cmp_hz": 45.0 if self.chiller_on else 0.0,
        }
        for r, plant in self.rooms.items():
            readings[f"{r}_temperature"] = plant.T_room
            readings[f"{r}_Q_fan_w"] = room_Q_kw[r] * 1000.0

        self.store.append(self.time, readings)
        return readings
