import logging
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from battery_ems.interfaces.influx import EnergyDataInterface
from battery_ems.interfaces.meter_config import load_meter_config

log = logging.getLogger(__name__)

# Clock StateReader anchors "now" to when reading the synthetic store.
# Defaults to real wall-clock time; run_synthetic_demo.py overrides this via
# set_clock() to drive reads from a SyntheticBuilding's own simulated clock
# instead -- otherwise a fast-forwarded run's synthetic data (timestamped
# ahead of real time) would never satisfy `index <= target_ts` and every
# read would return NaN. (forecast_provider.py's weather/tariff scheduling
# intentionally still uses real wall-clock time -- DWD is a live external
# service anchored to the real clock regardless of simulated drift, and
# tariff/comfort-hour logic only depends on hour-of-day, which staying
# close to real time keeps sane.)
def _clock() -> datetime:
    return datetime.now(timezone.utc)


def set_clock(fn) -> None:
    global _clock
    _clock = fn


# Maps MPC room IDs to meter config keys in meters_demo.yaml
ROOM_TEMP_METERS = {
    "room_1": "demo_temperature_R1",
    "room_2": "demo_temperature_R2",
    "room_3": "demo_temperature_R3",
    "room_4": "demo_temperature_R4",
    "room_5": "demo_temperature_R5",
}

# Heat flow meters per room (Pwr field, kW). Sign convention: MPC expects
# Q_fan <= 0 (cooling = heat removed from room); the synthetic building
# already reports Pwr in that convention (see synthetic_building/plant.py).
ROOM_QFAN_METERS = {
    "room_1": "demo_Q_R1",
    "room_2": "demo_Q_R2",
    "room_3": "demo_Q_R3",
    "room_4": "demo_Q_R4",
    "room_5": "demo_Q_R5",
}


def _col_to_array(df: pd.DataFrame, col: str) -> np.ndarray:
    """Convert a DataFrame column to a float numpy array, NaN where missing."""
    if df.empty or col not in df.columns:
        return np.array([], dtype=float)
    return df[col].to_numpy(dtype=float)


def _last_closed_bucket_start(now: datetime, resolution_s: int) -> datetime:
    """
    Absolute start-time of the last FULLY CLOSED aggregateWindow-equivalent
    bucket of width resolution_s, as of `now`.
    """
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    seconds_since_epoch = (now - epoch).total_seconds()
    current_bucket_start_s = seconds_since_epoch - (seconds_since_epoch % resolution_s)
    current_bucket_start = epoch + timedelta(seconds=current_bucket_start_s)
    return current_bucket_start - timedelta(seconds=resolution_s)


def _at_time(df: pd.DataFrame, col: str, target_ts: datetime) -> float:
    """Return the most recent NON-NaN value at-or-before target_ts."""
    if df.empty or col not in df.columns:
        return float("nan")
    target_ts = pd.Timestamp(target_ts)
    eligible = df.loc[df.index <= target_ts, col].dropna()
    if len(eligible) == 0:
        return float("nan")
    return float(eligible.iloc[-1])


class StateReader:
    """
    Reads the current state of the synthetic demo building from the
    TimeSeriesStore, via the same EnergyDataInterface.get_meter_data() seam
    the private repo uses against real InfluxDB. Logic below is otherwise
    unchanged from the private repo's state_reader.py.
    """

    def __init__(self, meter_config_file: str = "meters_demo.yaml", db: EnergyDataInterface | None = None):
        # `db` override lets a demo entrypoint (run_synthetic_demo.py) share
        # the same in-memory TimeSeriesStore a SyntheticBuilding is writing
        # to, so newly-simulated readings are visible immediately rather
        # than only after a save()/reload() round-trip through disk.
        self.db = db if db is not None else EnergyDataInterface()
        self.config = load_meter_config(meter_config_file)

    def read(self) -> dict:
        T_amb, T_amb_lag = self._read_weather_temp()
        T_sup, T_sup_lag = self._read_T_sup()

        room_temps_history = {r: self._read_room_temp_history(r) for r in ROOM_TEMP_METERS}
        room_temps = {r: hist[-1] for r, hist in room_temps_history.items()}

        Q_fan_measured = {r: self._read_Q_fan(r) for r in ROOM_QFAN_METERS}
        chiller_running = self._read_chiller_running()
        battery_soc_kwh = self._read_battery_soc()

        return {
            "T_amb": T_amb,
            "T_amb_5min_ago": T_amb_lag,
            "T_sup": T_sup,
            "T_sup_5min_ago": T_sup_lag,
            "room_temps": room_temps,
            "room_temps_history": room_temps_history,
            "Q_fan_measured": Q_fan_measured,
            "chiller_running": chiller_running,
            "battery_soc_kwh": battery_soc_kwh,
        }

    def _read_battery_soc(self) -> float:
        now = _clock()
        current_ts = _last_closed_bucket_start(now, 30)
        df = self.db.get_meter_data(self.config["demo_battery_soc"], start="-8m", resolution="30s")
        return _at_time(df, "soc_kwh", current_ts)

    def _read_weather_temp(self) -> tuple[float, float]:
        now = _clock()
        current_ts = _last_closed_bucket_start(now, 30)
        lag_ts = current_ts - timedelta(minutes=5)
        df = self.db.get_meter_data(
            self.config["weather_station_temp"], start="-8m", resolution="30s"
        )
        return _at_time(df, "EnvTmp", current_ts), _at_time(df, "EnvTmp", lag_ts)

    def _read_T_sup(self) -> tuple[float, float]:
        now = _clock()
        current_ts = _last_closed_bucket_start(now, 30)
        lag_ts = current_ts - timedelta(minutes=5)
        df = self.db.get_meter_data(
            self.config["demo_Q_AC"], start="-8m", resolution="30s"
        )
        current = _at_time(df, "TmpSup", current_ts)
        lag = _at_time(df, "TmpSup", lag_ts)

        if math.isnan(current) or math.isnan(lag):
            df_chiller = self.db.get_meter_data(
                self.config["demo_chiller"], start="-8m", resolution="30s"
            )
            df_chiller = df_chiller.sort_index()
            if math.isnan(current):
                current = _at_time(df_chiller, "GH.OUT.ACTS_SupTmp", current_ts)
                if not math.isnan(current):
                    log.warning(f"T_sup (demo_Q_AC) NaN — using chiller's own SupTmp: {current:.1f}°C")
            if math.isnan(lag):
                lag = _at_time(df_chiller, "GH.OUT.ACTS_SupTmp", lag_ts)

        return current, lag

    def _read_room_temp_history(self, room: str, n: int = 3) -> list:
        now = _clock()
        current_ts = _last_closed_bucket_start(now, 30)
        key = ROOM_TEMP_METERS[room]
        df = self.db.get_meter_data(self.config[key], start="-20m", resolution="30s")
        anchors = [current_ts - timedelta(minutes=5 * i) for i in range(n - 1, -1, -1)]
        return [_at_time(df, "temperature", ts) for ts in anchors]

    def _read_Q_fan(self, room: str) -> float:
        now = _clock()
        current_ts = _last_closed_bucket_start(now, 300)
        key = ROOM_QFAN_METERS[room]
        df = self.db.get_meter_data(self.config[key], start="-20m", resolution="5m")
        pwr = _at_time(df, "Pwr", current_ts)
        if math.isnan(pwr):
            return float("nan")
        return pwr / 1000.0  # store stores W; model expects kW

    def _read_chiller_running(self) -> float:
        now = _clock()
        current_ts = _last_closed_bucket_start(now, 30)
        df = self.db.get_meter_data(self.config["demo_chiller"], start="-8m", resolution="30s")
        df = df.sort_index()
        cmp_hz = _at_time(df, "GH.OUT.ACTS_CmpHz", current_ts)
        if math.isnan(cmp_hz):
            return float("nan")
        return 1.0 if cmp_hz > 5.0 else 0.0

    def read_history(self, hours: int = 24) -> dict:
        start = f"-{hours}h"
        res = "5m"

        df_amb = self.db.get_meter_data(self.config["weather_station_temp"], start=start, resolution=res)
        df_sup = self.db.get_meter_data(self.config["demo_Q_AC"], start=start, resolution=res)
        df_solar = self.db.get_meter_data(self.config["weather_station_solar"], start=start, resolution=res)

        T_amb_arr = _col_to_array(df_amb, "EnvTmp")
        T_sup_arr = _col_to_array(df_sup, "TmpSup")
        Solar_arr = _col_to_array(df_solar, "GlobIrradHoriz") / 1000.0  # W/m² → kW/m²

        room_temps, Q_fan = {}, {}
        for room, key in ROOM_TEMP_METERS.items():
            df = self.db.get_meter_data(self.config[key], start=start, resolution=res)
            room_temps[room] = _col_to_array(df, "temperature")

        for room, key in ROOM_QFAN_METERS.items():
            df = self.db.get_meter_data(self.config[key], start=start, resolution=res)
            Q_fan[room] = _col_to_array(df, "Pwr") / 1000.0  # W → kW

        all_arrays = [T_amb_arr, T_sup_arr, Solar_arr] + list(room_temps.values()) + list(Q_fan.values())
        lengths = [len(a) for a in all_arrays if len(a) > 0]
        n = min(lengths) if lengths else 0

        return {
            "T_amb": T_amb_arr[:n],
            "T_sup": T_sup_arr[:n],
            "Solar": Solar_arr[:n],
            "room_temps": {r: a[:n] for r, a in room_temps.items()},
            "Q_fan": {r: a[:n] for r, a in Q_fan.items()},
        }


if __name__ == "__main__":
    reader = StateReader()
    state = reader.read()
    print(state)
