"""
Synthetic drop-in replacement for the private repo's InfluxDB-backed
`EnergyDataInterface`. Same public method signature
(`get_meter_data(meter_config, start, end, resolution) -> DataFrame`), so
every caller elsewhere in this codebase (state_reader.py, pv_load_reader.py)
works completely unmodified. Backed by `TimeSeriesStore`
(emulators/synthetic_building/store.py) instead of a live InfluxDB server --
there is no real building here, so there's nothing to connect to and no
`.env`/credentials needed.
"""
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from battery_ems.emulators.synthetic_building.store import TimeSeriesStore

DEFAULT_STORE_PATH = Path(__file__).parent.parent.parent.parent / "data" / "synthetic_store.parquet"

_DURATION_RE = re.compile(r"^-(\d+)(s|m|h|d)$")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}

# Overridable by run_synthetic_demo.py so relative windows ("-8m", "now()")
# resolve against a SyntheticBuilding's simulated clock during a
# fast-forwarded run instead of real wall-clock time -- see the matching
# hook in state_reader.py for why this matters.
def _clock() -> datetime:
    return datetime.now(timezone.utc)


def set_clock(fn) -> None:
    global _clock
    _clock = fn


def _resolve_time(spec: str, now: datetime) -> datetime:
    """Parses the small subset of Flux time syntax actually used in this
    codebase: 'now()', a relative duration like '-24h'/'-8m', or an ISO 8601
    absolute timestamp."""
    if spec == "now()":
        return now
    m = _DURATION_RE.match(spec)
    if m:
        n, unit = m.groups()
        return now - timedelta(seconds=int(n) * _UNIT_SECONDS[unit])
    ts = pd.Timestamp(spec)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime()


class EnergyDataInterface:
    """Synthetic stand-in for the real InfluxDB client -- same method
    signature, reads from a shared TimeSeriesStore instead."""

    def __init__(self, store: TimeSeriesStore | None = None):
        self.store = store if store is not None else TimeSeriesStore(DEFAULT_STORE_PATH)

    def get_meter_data(
            self,
            meter_config: dict,
            start: str = "-24h",
            end: str = "now()",
            resolution: str = "30s",
    ) -> pd.DataFrame:
        now = _clock()
        start_ts = pd.Timestamp(_resolve_time(start, now))
        end_ts = pd.Timestamp(_resolve_time(end, now))
        # pandas resample needs a pandas-style offset alias, not Flux's "30s"/"5m"
        pd_resolution = resolution.replace("m", "min") if resolution.endswith("m") else resolution

        meter_type = meter_config.get("type", "single")

        if meter_type == "single":
            df = self.store.query([meter_config["signal"]], start_ts, end_ts, pd_resolution)
            return df.rename(columns={meter_config["signal"]: meter_config["output_column"]})

        if meter_type == "chiller":
            cols = [meter_config["sup_tmp_signal"], meter_config["cmp_hz_signal"]]
            df = self.store.query(cols, start_ts, end_ts, pd_resolution)
            return df.rename(columns={
                meter_config["sup_tmp_signal"]: "GH.OUT.ACTS_SupTmp",
                meter_config["cmp_hz_signal"]: "GH.OUT.ACTS_CmpHz",
            })

        if meter_type == "power":
            df = self.store.query([meter_config["signal"]], start_ts, end_ts, pd_resolution)
            return df.rename(columns={meter_config["signal"]: "active_power"})

        raise ValueError(f"Unknown synthetic meter type: {meter_type!r}")
