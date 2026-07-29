"""
Fast-forward closed-loop demo: the real run_mpc.py control-step logic
(unmodified, imported directly), driven against a live SyntheticBuilding
with no wall-clock sleeps -- proves the full loop actually closes (a
command changes the simulated building's physics, which the next sensor
read then reflects), not just that a single solve succeeds.

Usage:
    python -m scripts.run_synthetic_demo [--cycles N]

Requires the store to already have some history (run
scripts/seed_synthetic_history.py first) so the observer doesn't need a
cold start, though it works from a cold start too (see run_mpc.py).
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import battery_ems.interfaces.influx as influx_module  # noqa: E402
import battery_ems.interfaces.PLC_API as plc_module  # noqa: E402
import battery_ems.mpc.control_writer as control_writer_module  # noqa: E402
import battery_ems.mpc.state_reader as state_reader_module  # noqa: E402
from battery_ems.controllers.gurobipy_mpc import build_parametric_mpc  # noqa: E402
from battery_ems.emulators.synthetic_building.plant import SyntheticBuilding  # noqa: E402
from battery_ems.emulators.synthetic_building.store import TimeSeriesStore  # noqa: E402
from battery_ems.interfaces.influx import EnergyDataInterface  # noqa: E402
from battery_ems.interfaces.PLC_API import PLC_API  # noqa: E402
from battery_ems.mpc.control_writer import ControlWriter  # noqa: E402
from battery_ems.mpc.forecast_provider import ForecastProvider  # noqa: E402
from battery_ems.mpc.rc_observer import RCObserver  # noqa: E402
from battery_ems.mpc.run_mpc import (  # noqa: E402
    HORIZON_STEPS, LOG_FILE, STATE_FILE, load_mpc_config, run_one_step,
)
from battery_ems.mpc.state_reader import StateReader  # noqa: E402
from battery_ems.mpc.step_logger import StepLogger  # noqa: E402

STORE_PATH = REPO / "data" / "synthetic_store.parquet"
BLOCK_SIZE = 3  # 5-min physics steps per 15-min control step


def main(cycles: int) -> None:
    # No wall-clock sleeps: shrink control_writer's real-hardware retry
    # delay (10s x 2 per write against real PLC hardware) to zero -- there's
    # nothing to physically wait for against the synthetic plant.
    control_writer_module.RETRY_DELAY_S = 0

    store = TimeSeriesStore(STORE_PATH)
    if len(store) == 0:
        print("WARNING: store is empty -- run scripts/seed_synthetic_history.py "
              "first for a realistic demo. Continuing with a cold start.")
        start_time = datetime.now(timezone.utc)
    else:
        start_time = store._df.index[-1].to_pydatetime()

    building = SyntheticBuilding(store, start_time=start_time, seed=7)
    plc_module.bind_building(building)

    # Route both the store-anchoring clock and the relative-window resolver
    # clock at the synthetic building's own simulated time, so reads
    # correctly see what's just been simulated (see state_reader.py /
    # influx.py's set_clock() docstrings for why this is needed for a
    # fast-forwarded run).
    state_reader_module.set_clock(lambda: building.time)
    influx_module.set_clock(lambda: building.time)

    print("Loading MPC config...")
    config = load_mpc_config()
    rc_models = config["rc_models"]

    print("Building parametric Gurobi model (once)...")
    m, params, vars_dict = build_parametric_mpc(HORIZON_STEPS, config)

    shared_db = EnergyDataInterface(store=store)
    state_reader = StateReader(meter_config_file="meters_demo.yaml", db=shared_db)
    observer = RCObserver(rc_models=rc_models, state_file=STATE_FILE)
    if observer.needs_warmup:
        observer.warmup_from_history(state_reader)

    forecast_provider = ForecastProvider()
    logger = StepLogger(log_file=LOG_FILE)
    fallback_state = {"consecutive_failures": 0, "last_plan": None, "predicted_next": None}

    plc_api = PLC_API(buildings=["demo"])
    control_writer = ControlWriter(plc_api=plc_api, building="demo")

    print(f"Starting fast-forward closed loop: {cycles} cycles of 15 simulated minutes each "
          f"(building clock starts at {building.time.isoformat()}).")

    T_room1_history = []
    for cycle in range(cycles):
        # Advance physics 15 sim-minutes (3 x 5-min steps) under whatever
        # was last commanded, THEN read/solve/write for the next interval --
        # mirrors real causality (a decision only takes effect going
        # forward, not retroactively).
        for _ in range(BLOCK_SIZE):
            building.step()

        print(f"\n--- Cycle {cycle + 1}/{cycles} | sim time {building.time.isoformat()} "
              f"| room_1={building.rooms['room_1'].T_room:.2f}°C | "
              f"chiller={'ON' if building.chiller_on else 'OFF'} ---")
        T_room1_history.append(building.rooms["room_1"].T_room)

        run_one_step(
            m, params, vars_dict,
            state_reader, observer, forecast_provider,
            control_writer, logger, rc_models,
            fallback_state, dry_run=False,
        )

    store.save()
    print(f"\nDone. {cycles} cycles simulated, store saved to {STORE_PATH}.")
    print(f"room_1 temperature trajectory (°C): {[round(t, 2) for t in T_room1_history]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=8, help="Number of 15-min control cycles to run.")
    args = parser.parse_args()
    main(args.cycles)
