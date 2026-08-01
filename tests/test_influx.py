"""
Tests for the synthetic EnergyDataInterface (interfaces/influx.py) -- the
drop-in replacement for the private repo's real InfluxDB-backed client.
Where the private repo's equivalent test mocks the InfluxDBClient to avoid a
real network call, there's nothing to mock here: TimeSeriesStore already IS
the fake backend, so these tests just exercise it directly, plus the
relative-time parsing and clock-injection (set_clock) that's this layer's
own addition, needed so a fast-forwarded run's synthetic timestamps still
resolve correctly (see influx.py / state_reader.py's set_clock docstrings).
"""
from datetime import datetime, timezone

import pandas as pd
import pytest

import battery_ems.interfaces.influx as influx_module
from battery_ems.emulators.synthetic_building.store import TimeSeriesStore
from battery_ems.interfaces.influx import EnergyDataInterface, _resolve_time


@pytest.fixture
def store_with_data(tmp_path):
    store = TimeSeriesStore(tmp_path / "store.parquet")
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    for i in range(4):
        store.append(base + pd.Timedelta(minutes=5 * i), {
            "EnvTmp": 20.0 + i,
            "T_sup": 10.0 + i,
            "chiller_cmp_hz": 45.0 if i >= 2 else 0.0,
            "pv_generation_kw": 1.0 + i,
        })
    return store


@pytest.fixture(autouse=True)
def reset_clock():
    """influx.py's clock is a module-level global (set_clock) -- reset it
    after every test so an override doesn't leak into unrelated tests."""
    yield
    influx_module.set_clock(lambda: datetime.now(timezone.utc))


def _query(db, meter, start="2026-01-01T00:00:00Z", end="2026-01-01T00:20:00Z", resolution="5m"):
    return db.get_meter_data(meter, start=start, end=end, resolution=resolution)


def test_single_meter_renames_signal_to_matching_output_column(store_with_data):
    db = EnergyDataInterface(store=store_with_data)
    meter = {"type": "single", "signal": "EnvTmp", "output_column": "EnvTmp"}
    df = _query(db, meter)
    assert "EnvTmp" in df.columns
    assert df["EnvTmp"].iloc[0] == pytest.approx(20.0)


def test_single_meter_renames_signal_to_a_different_output_column(store_with_data):
    db = EnergyDataInterface(store=store_with_data)
    meter = {"type": "single", "signal": "pv_generation_kw", "output_column": "active_power"}
    df = _query(db, meter)
    assert "active_power" in df.columns
    assert "pv_generation_kw" not in df.columns


def test_chiller_meter_returns_both_readback_columns(store_with_data):
    db = EnergyDataInterface(store=store_with_data)
    meter = {"type": "chiller", "sup_tmp_signal": "T_sup", "cmp_hz_signal": "chiller_cmp_hz"}
    df = _query(db, meter)
    assert "GH.OUT.ACTS_SupTmp" in df.columns
    assert "GH.OUT.ACTS_CmpHz" in df.columns
    assert df["GH.OUT.ACTS_CmpHz"].iloc[-1] > 5.0  # last two rows have chiller_cmp_hz=45


def test_power_meter_renames_signal_to_active_power(store_with_data):
    db = EnergyDataInterface(store=store_with_data)
    meter = {"type": "power", "signal": "pv_generation_kw"}
    df = _query(db, meter)
    assert "active_power" in df.columns


def test_unknown_meter_type_raises(tmp_path):
    db = EnergyDataInterface(store=TimeSeriesStore(tmp_path / "store.parquet"))
    with pytest.raises(ValueError):
        _query(db, {"type": "nonexistent"})


def test_query_outside_data_range_returns_nan_not_empty(store_with_data):
    db = EnergyDataInterface(store=store_with_data)
    meter = {"type": "single", "signal": "EnvTmp", "output_column": "EnvTmp"}
    df = _query(db, meter, start="2020-01-01T00:00:00Z", end="2020-01-01T01:00:00Z")
    assert df.empty or df["EnvTmp"].isna().all()


# --- relative-time parsing (_resolve_time) ----------------------------------

def test_resolve_time_now():
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert _resolve_time("now()", now) == now


def test_resolve_time_relative_minutes():
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    resolved = _resolve_time("-8m", now)
    assert (now - resolved).total_seconds() == 8 * 60


def test_resolve_time_relative_hours():
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    resolved = _resolve_time("-24h", now)
    assert (now - resolved).total_seconds() == 24 * 3600


def test_resolve_time_absolute_iso():
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    resolved = _resolve_time("2026-01-01T00:00:00Z", now)
    assert resolved == datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


# --- clock injection (set_clock) --------------------------------------------

def test_set_clock_overrides_relative_time_resolution(store_with_data):
    """A fast-forwarded run (run_synthetic_demo.py) needs relative windows
    ("-8m", "now()") to resolve against the SyntheticBuilding's own
    simulated clock, not real wall-clock time -- otherwise a run whose
    simulated timestamps have drifted from real time would see every query
    return NaN (nothing "recent" relative to the real clock)."""
    fake_now = pd.Timestamp("2026-01-01T00:15:00Z").to_pydatetime()
    influx_module.set_clock(lambda: fake_now)

    db = EnergyDataInterface(store=store_with_data)
    meter = {"type": "single", "signal": "EnvTmp", "output_column": "EnvTmp"}
    df = db.get_meter_data(meter, start="-20m", end="now()", resolution="5m")

    assert not df.empty
    assert df["EnvTmp"].iloc[-1] == pytest.approx(23.0)  # the 00:15 sample (20+3)
