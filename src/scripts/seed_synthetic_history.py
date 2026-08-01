"""
Creates synthetic TimeSeriesStore with plausible "historical" data by
running SyntheticBuilding forward under a bang-bang
controller.

Usage:
    python -m scripts.seed_synthetic_history [--days N]
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from battery_ems.emulators.synthetic_building.plant import (
    SyntheticBuilding,
)
from battery_ems.emulators.synthetic_building.store import TimeSeriesStore

STORE_PATH = REPO / "data" / "synthetic_store.parquet"
T_TARGET = 23.0
DEADBAND = 1.0


def run_reactive_baseline(building: SyntheticBuilding, n_steps: int) -> None:
    """Simple bang-bang controller: cool a room once it's DEADBAND above
    target, run the chiller whenever any room wants cooling."""
    for i in range(n_steps):
        need_cooling = {r: p.T_room > T_TARGET + DEADBAND for r, p in building.rooms.items()}
        any_cooling = any(need_cooling.values())
        building.set_supply_setpoint(8.0 if any_cooling else 20.0)
        for r in building.rooms:
            building.set_fan_speed(r, "normal" if need_cooling[r] else "off")
        building.step()
        if (i + 1) % 576 == 0:  # every 2 simulated days (5-min steps)
            print(f"  seeded {i + 1}/{n_steps} steps ({building.time.isoformat()})")


def main(days: int) -> None:
    n_steps = days * 24 * 12  # 5-min steps
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    store = TimeSeriesStore(STORE_PATH)
    building = SyntheticBuilding(store, start_time=start_time, seed=42)

    print(f"Seeding {days} days ({n_steps} steps) of synthetic history, "
          f"{start_time.isoformat()} -> {end_time.isoformat()}...")
    run_reactive_baseline(building, n_steps)

    store.save()
    print(f"Saved {len(store)} rows to {STORE_PATH}")
    print(f"Building clock now at: {building.time.isoformat()} (real now: {datetime.now(timezone.utc).isoformat()})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14)
    args = parser.parse_args()
    main(args.days)
