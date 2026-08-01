"""
Joint cooling+PV+battery MPC control loop for the synthetic demo building.

Usage:
    python -m battery_ems.mpc.run_mpc            # live control (writes to the synthetic PLC every 15 min)
    python -m battery_ems.mpc.run_mpc --dry-run  # one solve, no PLC, saves plan plots to figures/

This module is just the entry point: load config, build the MPC model once,
build the control loop, then either run one dry-run step or the scheduled
15-min live loop. The per-step pipeline itself (sensor read -> observer
update -> forecast -> solve -> apply decision -> log) lives in
mpc/control_loop.py.
"""
import argparse
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from battery_ems.controllers.gurobipy_mpc import ParametricMPC
from battery_ems.interfaces.plc_api import PLC_API
from battery_ems.mpc.control_loop import MPCControlLoop
from battery_ems.mpc.control_writer import ControlWriter
from battery_ems.mpc.forecast_provider import ForecastProvider
from battery_ems.mpc.pv_load_forecast import PVLoadForecastProvider
from battery_ems.mpc.rc_observer import RCObserver
from battery_ems.mpc.state_reader import StateReader
from battery_ems.mpc.step_logger import StepLogger

BLUE  = "\033[34m"
RESET = "\033[0m"

class _BlueFormatter(logging.Formatter):
    def format(self, record):
        return BLUE + super().format(record) + RESET

ROOT = Path(__file__).parent.parent.parent.parent  # mpc/ -> battery_ems/ -> src/ -> project root
CONSOLE_LOG_FILE = ROOT / "data" / "mpc_logs" / "run_mpc.log"

log = logging.getLogger(__name__)


def setup_logging() -> None:
    """Blue console + plain file handler, both at INFO. Kept out of module scope
    (unlike a bare logging.basicConfig() call at import time) so importing this
    module -- e.g. run_synthetic_demo.py pulling in MPCControlLoop -- doesn't
    silently mutate global logging config as a side effect; callers that want
    this behavior call it explicitly."""
    CONSOLE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(_BlueFormatter("%(asctime)s %(levelname)s %(message)s"))
    file_handler = logging.FileHandler(CONSOLE_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])


CONFIG_DIR = ROOT / "src" / "battery_ems" / "controllers"
STATE_FILE = ROOT / "data" / "mpc_state" / "observer_state.json"
LOG_FILE = ROOT / "data" / "mpc_logs" / "mpc_steps.jsonl"

HORIZON_STEPS = 288   # 24 h at 5-min resolution
BUILDING = "demo"


def load_mpc_config() -> dict:
    with open(CONFIG_DIR / "mpc_plant_power_coefs.json") as f:
        plant_physics = json.load(f)
    with open(CONFIG_DIR / "mpc_unified_fan_coefficients.json") as f:
        fan_physics = json.load(f)
    with open(CONFIG_DIR / "mpc_rc_models.json") as f:
        rc_models = json.load(f)
    with open(CONFIG_DIR / "mpc_battery_coefs.json") as f:
        battery_coefs = json.load(f)
    return {
        "plant_physics": plant_physics,
        "fan_physics": fan_physics,
        "rc_models": rc_models,
        "battery_physics": battery_coefs["battery_demo"],
    }


DRY_RUN_DEFAULT = True


def _seconds_to_next_15min() -> float:
    now = datetime.now(timezone.utc)
    seconds_into_block = (now.minute % 15) * 60 + now.second + now.microsecond / 1e6
    return math.ceil(900 - seconds_into_block)


def main(dry_run: bool, immediate: bool = False, mode_end: datetime | None = None) -> None:
    log.info("Loading MPC config...")
    config = load_mpc_config()
    rc_models = config["rc_models"]
    battery_capacity_kwh = config["battery_physics"]["capacity_kwh"]

    log.info("Building parametric joint MPC model (once)...")
    mpc = ParametricMPC(HORIZON_STEPS, config)

    log.info("Initializing state reader and observer...")
    state_reader = StateReader(meter_config_file="meters_demo.yaml")
    observer = RCObserver(rc_models=rc_models, state_file=STATE_FILE)
    if observer.needs_warmup:
        observer.warmup_from_history(state_reader)

    loop = MPCControlLoop(
        mpc=mpc,
        state_reader=state_reader,
        observer=observer,
        forecast_provider=ForecastProvider(),
        pv_load_forecast_provider=PVLoadForecastProvider(),
        logger=StepLogger(log_file=LOG_FILE),
        rc_models=rc_models,
        battery_capacity_kwh=battery_capacity_kwh,
    )

    if mode_end is not None:
        log.info(f"mode_end={mode_end.isoformat()} -- comfort-schedule tightening will be "
                  f"suppressed at/after this time (handing off to another controller).")

    if dry_run:
        log.info("=== DRY RUN — no PLC connection, no writes ===")
        loop.run_one_step(control_writer=None, dry_run=True, mode_end=mode_end)
        return

    # Live loop
    plc_api = PLC_API(buildings=[BUILDING])
    control_writer = ControlWriter(plc_api=plc_api, building=BUILDING)

    if immediate:
        log.info("Running first step immediately (--immediate), then aligning to 15-min boundaries.")
    else:
        wait = _seconds_to_next_15min()
        log.info(f"Aligning to 15-min boundary — first step in {wait}s.")
        time.sleep(wait)

    log.info("Starting control loop (15-min steps).")
    while True:
        try:
            loop.run_one_step(control_writer, mode_end=mode_end)
        except Exception:
            log.exception("Unhandled error in control step")

        wait = _seconds_to_next_15min()
        log.info(f"Next step in {wait}s.")
        time.sleep(wait)


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", dest="dry_run", action="store_true", default=None,
        help="One solve, no PLC connection, no writes -- saves plan plots to figures/ and exits."
    )
    parser.add_argument(
        "--live", dest="dry_run", action="store_false",
        help="Force live control (writes to the synthetic PLC), overriding DRY_RUN_DEFAULT."
    )
    parser.add_argument(
        "--immediate", action="store_true",
        help="Run the first control step immediately instead of waiting for the next 15-min boundary."
    )
    parser.add_argument(
        "--mode-end", dest="mode_end", default=None,
        help="ISO8601 timestamp marking when this MPC instance hands off to another controller."
    )
    args = parser.parse_args()
    dry_run = DRY_RUN_DEFAULT if args.dry_run is None else args.dry_run
    mode_end = datetime.fromisoformat(args.mode_end) if args.mode_end else None
    main(dry_run=dry_run, immediate=args.immediate, mode_end=mode_end)
