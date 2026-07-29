import numpy as np
import pandas as pd
import pytest

from battery_ems.emulators.synthetic_building.rc_plant import (
    ROOM_PARAMS, RCPlant, discretize,
)
from battery_ems.emulators.synthetic_building.store import TimeSeriesStore


def test_discretize_is_stable_for_every_room():
    for room, params in ROOM_PARAMS.items():
        model = discretize(params)
        A = np.array(model["A"])
        eigvals = np.linalg.eigvals(A)
        assert np.all(np.abs(eigvals) < 1.0), f"{room}: unstable discrete A (eig={eigvals})"
        assert np.all(np.abs(eigvals) > 0.5), f"{room}: implausibly fast dynamics (eig={eigvals})"


def test_discretize_shapes_match_deployed_json_convention():
    model = discretize(ROOM_PARAMS["room_1"])
    assert len(model["A"]) == 2 and len(model["A"][0]) == 2
    assert len(model["B"]) == 2 and len(model["B"][0]) == 4
    assert model["B"][0][3] == 0.0 and model["B"][1][3] == 0.0  # unused 4th input
    assert model["C"] == pytest.approx([1.0, 0.0])
    assert model["D"] == [0.0, 0.0, 0.0, 0.0]


def test_plant_cools_when_fan_active():
    plant = RCPlant(ROOM_PARAMS["room_3"], x0=(26.0, 26.0))
    T0 = plant.T_room
    for _ in range(24):  # 2 hours of steady active cooling
        plant.step(T_amb=30.0, Q_fan_kw=-1.5, Solar_kw_m2=0.0)
    assert plant.T_room < T0, "room should cool under sustained negative Q_fan"


def test_plant_warms_toward_ambient_with_no_cooling():
    plant = RCPlant(ROOM_PARAMS["room_1"], x0=(18.0, 18.0))
    for _ in range(48):  # 4 hours passive
        plant.step(T_amb=32.0, Q_fan_kw=0.0, Solar_kw_m2=0.0)
    assert plant.T_room > 18.0, "room should warm toward a hot ambient with no active cooling"


def test_store_range_query_and_resample(tmp_path):
    store = TimeSeriesStore(tmp_path / "store.parquet")
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    for i in range(12):  # 1 hour at 5-min cadence
        store.append(base + pd.Timedelta(minutes=5 * i), {"temperature": 20.0 + i * 0.1})

    out = store.query(["temperature"], base, base + pd.Timedelta(hours=1), resolution="15min")
    # samples only span [00:00, 00:55], so resample yields 4 bins: 00:00/15/30/45
    assert len(out) == 4
    assert out.index.tz is not None
    assert out["temperature"].iloc[0] == pytest.approx(20.1, abs=0.05)  # mean of first 3 samples


def test_store_persists_across_reload(tmp_path):
    path = tmp_path / "store.parquet"
    store = TimeSeriesStore(path)
    ts = pd.Timestamp("2026-01-01T00:00:00Z")
    store.append(ts, {"EnvTmp": 21.5})
    store.save()

    reloaded = TimeSeriesStore(path)
    assert len(reloaded) == 1
    assert reloaded.latest(["EnvTmp"])["EnvTmp"] == pytest.approx(21.5)


def test_store_latest_missing_column_is_nan(tmp_path):
    store = TimeSeriesStore(tmp_path / "store.parquet")
    assert np.isnan(store.latest(["nonexistent"])["nonexistent"])
