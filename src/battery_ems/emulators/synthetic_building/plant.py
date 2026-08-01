"""
SyntheticBuilding: the "digital twin" ground-truth building. Owns 5 rooms'
RCPlant instances, a chiller with its own T_sup response and simple diurnal
weather/PV-irradiance generation.

Advances one 5-min physics step at a time via step(), writing the resulting
readings into a shared TimeSeriesStore.
"""
import math
from datetime import datetime, timedelta

import numpy as np

from battery_ems.emulators.battery.battery import Battery
from battery_ems.emulators.synthetic_building.rc_plant import ROOM_PARAMS, RCPlant

DT_SECONDS = 300
DT_HOURS = DT_SECONDS / 3600.0

# Ground-truth PV/battery physics (independent of the MPC's own fitted
# belief in controllers/mpc_battery_coefs.json, same "ground truth vs
# belief" separation as the chiller/fan/RC constants above).
PV_CAPACITY_KW = 5.0
BATTERY_CAPACITY_KWH = 6.0
BATTERY_MAX_CHARGE_KW = 4.5
BATTERY_MAX_DISCHARGE_KW = 4.5
BATTERY_EFF_CHARGE = 0.95
BATTERY_EFF_DISCHARGE = 0.95

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


def _diurnal_weather(t: datetime, rng: np.random.Generator, T_amb_offset: float = 0.0) -> tuple[float, float]:
    """Simple plausible diurnal ambient temp (°C) + solar irradiance (W/m²).
    T_amb_offset shifts the whole diurnal band -- e.g. +10 gives a 28-40°C
    heat-wave scenario, hot enough that the chiller has to run to hold the
    22-24°C daytime comfort band (see forecast_provider.py's comfort
    schedule), instead of the default 18-30°C band that rarely forces it."""
    hour = t.hour + t.minute / 60.0
    T_amb = 24.0 + T_amb_offset + 6.0 * math.sin(2 * math.pi * (hour - 8) / 24) + rng.normal(0, 0.3)
    if 6 <= hour <= 19:
        solar = max(0.0, 700.0 * math.sin(math.pi * (hour - 6) / 13)) + rng.normal(0, 15)
        solar = max(0.0, solar)
    else:
        solar = 0.0
    return T_amb, solar


def _diurnal_pv_load(t: datetime, solar_wm2: float, rng: np.random.Generator) -> tuple[float, float]:
    """Ground-truth PV generation (kW, tracks irradiance) + uncontrollable
    household load (kW, base draw + evening bump), both plausible but not
    derived from any real building."""
    pv_kw = max(0.0, PV_CAPACITY_KW * (solar_wm2 / 700.0) + rng.normal(0, 0.05))
    hour = t.hour + t.minute / 60.0
    evening_bump = 0.6 * math.exp(-0.5 * ((hour - 19.5) / 2.0) ** 2)  # dinner/evening peak
    load_kw = max(0.05, 0.35 + evening_bump + rng.normal(0, 0.05))
    return pv_kw, load_kw


class SyntheticBuilding:
    def __init__(self, store, start_time: datetime, seed: int = 42, T_amb_offset: float = 0.0):
        self.store = store
        self.time = start_time
        self.rng = np.random.default_rng(seed)
        self.T_amb_offset = T_amb_offset
        self.rooms = {r: RCPlant(p, x0=(23.0, 23.0)) for r, p in ROOM_PARAMS.items()}
        self.T_sup = 18.0
        self._commanded_setpoint = 20.0  # starts "off"
        self.fan_speed: dict[str, str] = {r: "off" for r in self.rooms}
        self.battery = Battery(
            capacity_kwh=BATTERY_CAPACITY_KWH,
            power_charge_max_kw=BATTERY_MAX_CHARGE_KW,
            power_discharge_max_kw=BATTERY_MAX_DISCHARGE_KW,
            efficiency_charge=BATTERY_EFF_CHARGE,
            efficiency_discharge=BATTERY_EFF_DISCHARGE,
            initial_soc_kwh=BATTERY_CAPACITY_KWH * 0.5,
        )
        self._commanded_battery_power_kw = 0.0  # +discharge, -charge (matches gurobipy_mpc.py's convention)

    # -- PLC-facing interface (called by the synthetic PLC_API) -----------

    def set_supply_setpoint(self, setpoint: float) -> None:
        self._commanded_setpoint = setpoint

    def set_fan_speed(self, room: str, speed: str) -> None:
        self.fan_speed[room] = speed

    def reset_all_fans(self, speed: str) -> None:
        for r in self.rooms:
            self.fan_speed[r] = speed

    def set_battery_power(self, power_kw: float) -> None:
        """power_kw > 0 discharges, < 0 charges (matches gurobipy_mpc.py's
        Battery_Power_kW = P_discharge - P_charge convention)."""
        self._commanded_battery_power_kw = max(
            -BATTERY_MAX_CHARGE_KW, min(BATTERY_MAX_DISCHARGE_KW, power_kw)
        )

    # -- Physics ------------------------------------------------------------

    @property
    def chiller_on(self) -> bool:
        return self._commanded_setpoint <= CHILLER_COMMAND_THRESHOLD_C

    def step(self) -> dict:
        self.time = self.time + timedelta(seconds=DT_SECONDS)
        T_amb, solar_wm2 = _diurnal_weather(self.time, self.rng, self.T_amb_offset)
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

        pv_kw, load_kw = _diurnal_pv_load(self.time, solar_wm2, self.rng)
        # Battery.step()'s own convention is +charging; ours (matching
        # gurobipy_mpc.py) is +discharging -- negate at the boundary.
        self.battery.step(power_kw=-self._commanded_battery_power_kw, delta_t_hours=DT_HOURS)

        readings = {
            "EnvTmp": T_amb,
            "GlobIrradHoriz": solar_wm2,
            "T_sup": self.T_sup,
            "chiller_cmp_hz": 45.0 if self.chiller_on else 0.0,
            "pv_generation_kw": pv_kw,
            "household_load_kw": load_kw,
            "battery_soc_kwh": self.battery.soc_kwh,
        }
        for r, plant in self.rooms.items():
            readings[f"{r}_temperature"] = plant.T_room
            readings[f"{r}_Q_fan_w"] = room_Q_kw[r] * 1000.0

        self.store.append(self.time, readings)
        return readings
