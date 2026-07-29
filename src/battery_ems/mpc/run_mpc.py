"""
MPC control loop for the synthetic demo building.

Usage:
    python -m battery_ems.mpc.run_mpc            # live control (writes to the synthetic PLC every 15 min)
    python -m battery_ems.mpc.run_mpc --dry-run  # one solve, no PLC, saves plan plot to figures/

Step sequence each iteration:
    1. Read sensors from the synthetic store (via EnergyDataInterface)
    2. Fill any NaN gaps with observer / last-known fallbacks
    3. Update Kalman observer (3 × 5-min sub-steps)
    4. Fetch 24-h weather + price forecasts
    5. Solve MPC (Gurobi)
    6. Write decisions to the synthetic PLC (skipped in dry-run)
    7. Log step to JSONL

Identical control-loop logic to the private repo this is a sanitized demo
of -- only the I/O seams (StateReader/EnergyDataInterface, PLC_API) are
synthetic; see README.md.
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
import numpy as np

from battery_ems.controllers.gurobipy_mpc import (
    HOLDOVER_BLOCKS, build_parametric_mpc, plot_mpc_results, step_mpc,
)
from battery_ems.interfaces.PLC_API import PLC_API
from battery_ems.mpc.control_writer import ControlWriter
from battery_ems.mpc.forecast_provider import ForecastProvider
from battery_ems.mpc.prediction_writer import write_predictions
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
CONSOLE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_BlueFormatter("%(asctime)s %(levelname)s %(message)s"))
_file_handler = logging.FileHandler(CONSOLE_LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_stream_handler, _file_handler])
log = logging.getLogger(__name__)

CONFIG_DIR = ROOT / "src" / "battery_ems" / "controllers"
STATE_FILE = ROOT / "data" / "mpc_state" / "observer_state.json"
LOG_FILE = ROOT / "data" / "mpc_logs" / "mpc_steps.jsonl"
FIGURES_DIR = ROOT / "figures"

HORIZON_STEPS = 288   # 24 h at 5-min resolution
BLOCK_SIZE = 3        # 5-min physics steps per 15-min control step (run_one_step cadence)
BUILDING = "demo"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_mpc_config() -> dict:
    with open(CONFIG_DIR / "mpc_plant_power_coefs.json") as f:
        plant_physics = json.load(f)
    with open(CONFIG_DIR / "mpc_unified_fan_coefficients.json") as f:
        fan_physics = json.load(f)
    with open(CONFIG_DIR / "mpc_rc_models.json") as f:
        rc_models = json.load(f)
    return {"plant_physics": plant_physics, "fan_physics": fan_physics, "rc_models": rc_models}


# ---------------------------------------------------------------------------
# Sensor gap filling
# ---------------------------------------------------------------------------

def fill_sensor_gaps(measurements: dict, observer: RCObserver, rc_models: dict,
                      fallback_state: dict) -> dict:
    """
    Replace NaN sensor readings with the MPC's own prediction from the last
    successful solve where one exists, falling back further to
    observer/last-known values only if no prior solve is available yet.
    """
    state = observer._state
    predicted_list = fallback_state.get("predicted_next")
    idx = fallback_state.get("consecutive_failures", 0)
    predicted = predicted_list[idx] if predicted_list is not None and idx < len(predicted_list) else None

    if math.isnan(measurements["T_amb"]):
        if predicted is not None:
            measurements["T_amb"] = predicted["T_amb"]
            log.warning(f"T_amb NaN — using previous solve's forecast: {measurements['T_amb']:.1f}°C")
        else:
            measurements["T_amb"] = 20.0
            log.warning(f"T_amb NaN — no prior solve yet, flat fallback: {measurements['T_amb']:.1f}°C")

    if math.isnan(measurements["T_amb_5min_ago"]):
        measurements["T_amb_5min_ago"] = measurements["T_amb"]
        log.warning("T_amb_5min_ago NaN — using current T_amb")

    if math.isnan(measurements["T_sup"]):
        if predicted is not None:
            measurements["T_sup"] = predicted["T_sup"]
            log.warning(f"T_sup NaN — using previous solve's prediction: {measurements['T_sup']:.1f}°C")
        else:
            measurements["T_sup"] = state.get("T_sup_last", 16.0)
            log.warning(f"T_sup NaN — no prior solve yet, last-known fallback: {measurements['T_sup']:.1f}°C")

    if math.isnan(measurements["T_sup_5min_ago"]):
        measurements["T_sup_5min_ago"] = measurements["T_sup"]
        log.warning("T_sup_5min_ago NaN — using current T_sup")

    for r, model in rc_models.items():
        if math.isnan(measurements["room_temps"].get(r, float("nan"))):
            C = np.atleast_2d(np.array(model["C"]))
            x = np.array(observer.x_state[r])
            measurements["room_temps"][r] = float((C @ x).item())
            log.warning(f"{r} room temp NaN — model prediction: {measurements['room_temps'][r]:.1f}°C")

        history = measurements["room_temps_history"].get(r, [])
        n_nan = sum(1 for v in history if math.isnan(v))
        if n_nan:
            log.warning(
                f"{r} room_temps_history has {n_nan}/{len(history)} NaN sub-step(s) this cycle "
                f"— observer will run open-loop (no Kalman correction) for those."
            )

        if math.isnan(measurements["Q_fan_measured"].get(r, float("nan"))):
            if predicted is not None:
                measurements["Q_fan_measured"][r] = predicted["Q_fan"][r]
                log.warning(f"{r} Q_fan NaN — using previous solve's prediction: "
                            f"{measurements['Q_fan_measured'][r]:.3f} kW")
            else:
                measurements["Q_fan_measured"][r] = 0.0
                log.warning(f"{r} Q_fan NaN — no prior solve yet, substituting 0.0 kW")

    if math.isnan(measurements["chiller_running"]):
        if fallback_state.get("last_plan") is not None:
            measurements["chiller_running"] = float(fallback_state["last_plan"]["Chiller_Command"])
            log.warning(
                f"chiller_running NaN — falling back to last commanded state: "
                f"{'ON' if measurements['chiller_running'] else 'OFF'}"
            )
        else:
            measurements["chiller_running"] = 0.0
            log.warning("chiller_running NaN — no prior solve yet, conservative fallback: OFF")

    return measurements


# ---------------------------------------------------------------------------
# Infeasibility handling
# ---------------------------------------------------------------------------

def _handle_infeasible(
    fallback_state: dict, control_writer, dry_run: bool,
    timestamp, measurements, forecasts, observer, solve_time, mip_gap, logger,
) -> None:
    fallback_state["consecutive_failures"] += 1
    n = fallback_state["consecutive_failures"]

    if n <= HOLDOVER_BLOCKS and fallback_state["last_plan"] is not None:
        holdover = {
            "Chiller_Command": fallback_state["last_plan"]["Holdover_Chiller_Commands"][n - 1],
            "Fan_Commands":    fallback_state["last_plan"]["Holdover_Fan_Commands"][n - 1],
        }
        log.warning(
            f"MPC infeasible (#{n}/{HOLDOVER_BLOCKS}) — holdover block k={n} of last plan: "
            f"chiller={'ON' if holdover['Chiller_Command'] else 'OFF'}, "
            f"fans={holdover['Fan_Commands']}"
        )
        written = None
        if not dry_run:
            try:
                written = control_writer.write(holdover)
            except Exception as e:
                log.error(f"PLC holdover write failed: {e}")
        logger.log(timestamp, measurements, forecasts, holdover, written,
                   observer.x_state, solve_time, mip_gap, "infeasible_holdover")
    else:
        log.error(f"MPC infeasible (#{n}) — hard fallback: supply=10°C, rooms=22°C.")
        written = None
        if not dry_run:
            try:
                written = control_writer.write_hard_fallback()
            except Exception as e:
                log.error(f"PLC hard-fallback write failed: {e}")
        logger.log(timestamp, measurements, forecasts, None, written,
                   observer.x_state, solve_time, mip_gap, "infeasible_hard_fallback")


# ---------------------------------------------------------------------------
# Dry-run plot
# ---------------------------------------------------------------------------

def _save_plan_plot(vars_dict, params, forecasts, timestamp: datetime) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    out = FIGURES_DIR / f"dry_run_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
    plot_mpc_results(vars_dict, params, forecasts, HORIZON_STEPS, save_path=out)
    log.info(f"[DRY RUN] Plan plot saved: {out}")


# ---------------------------------------------------------------------------
# Single control step
# ---------------------------------------------------------------------------

def run_one_step(
    m, params, vars_dict,
    state_reader: StateReader,
    observer: RCObserver,
    forecast_provider: ForecastProvider,
    control_writer,           # ControlWriter | None in dry-run
    logger: StepLogger,
    rc_models: dict,
    fallback_state: dict,
    dry_run: bool = False,
    mode_end: datetime | None = None,
) -> None:
    timestamp = datetime.now(timezone.utc)
    rooms = list(rc_models.keys())

    # 1. Read sensors & fill NaN gaps
    try:
        measurements = state_reader.read()
    except Exception as e:
        log.error(f"Sensor read failed: {e}. Skipping step.")
        return
    measurements = fill_sensor_gaps(measurements, observer, rc_models, fallback_state)

    T_amb = measurements["T_amb"]
    T_sup = measurements["T_sup"]
    T_lift_historical = measurements["T_amb_5min_ago"] - measurements["T_sup_5min_ago"]

    # 2. Update Kalman observer
    u_prev_stored = observer.get_u_prev(rooms, T_amb_fallback=T_amb)
    u_prev = {
        r: [u_prev_stored[r][0], measurements["Q_fan_measured"][r], u_prev_stored[r][2], 0.0]
        for r in rooms
    }
    observer.update(
        y_measured=measurements["room_temps_history"],
        u_prev=u_prev,
        T_amb=T_amb,
        T_sup=T_sup,
    )
    initial_states = {
        "T_sup_current": T_sup,
        "Temp_Lift_historical": T_lift_historical,
        "x_state_current": observer.x_state,
        "Chiller_On_prev": measurements["chiller_running"],
    }

    # 3. Forecasts
    forecasts = forecast_provider.get(HORIZON_STEPS, T_amb_current=T_amb, rooms=rooms, mode_end=mode_end)

    # 4. Solve MPC
    t0 = time.time()
    try:
        optimal_action = step_mpc(m, params, vars_dict, initial_states, forecasts, HORIZON_STEPS)
        solve_time = time.time() - t0
        mip_gap = m.MIPGap if m.SolCount > 0 else None
        solver_status = {2: "optimal", 3: "infeasible", 5: "unbounded", 9: "time_limit"}.get(
            m.Status, f"status_{m.Status}"
        )
    except Exception as e:
        log.error(f"MPC solve exception: {e}")
        logger.log(timestamp, measurements, forecasts, None, None,
                   observer.x_state, time.time() - t0, None, "exception")
        return

    gap_str = f"{mip_gap:.4f}" if mip_gap is not None else "N/A"
    log.info(f"Solve: {solver_status}, gap={gap_str}, t={solve_time:.1f}s")

    if optimal_action is None:
        _handle_infeasible(
            fallback_state, control_writer, dry_run,
            timestamp, measurements, forecasts, observer, solve_time, mip_gap, logger,
        )
        return

    fallback_state["consecutive_failures"] = 0
    fallback_state["last_plan"] = optimal_action

    fallback_state["predicted_next"] = [
        {
            "T_sup": vars_dict["T_sup"][k * BLOCK_SIZE].X,
            "T_amb": forecasts["T_amb"][k * BLOCK_SIZE],
            "Q_fan": {r: vars_dict["Q_fan"][r][k * BLOCK_SIZE - 1].X for r in rooms},
        }
        for k in range(1, HOLDOVER_BLOCKS + 1)
    ]

    log.info(
        f"Decision: chiller={'ON' if optimal_action['Chiller_Command'] else 'OFF'}, "
        f"fans={optimal_action['Fan_Commands']}"
    )

    if not dry_run:
        try:
            write_predictions(timestamp, forecasts, vars_dict, rooms, HORIZON_STEPS)
        except Exception as e:
            log.warning(f"Writing predictions failed: {e}")

    # 5. Write to PLC (or save plan plot in dry-run)
    if dry_run:
        log.info("[DRY RUN] Skipping PLC write.")
        written = None
        _save_plan_plot(vars_dict, params, forecasts, timestamp)
    else:
        try:
            written = control_writer.write(optimal_action)
        except Exception as e:
            log.error(f"PLC write failed: {e}")
            written = None

    Q_fan_planned = {r: vars_dict["Q_fan"][r][0].X for r in rooms}
    u_current = {r: [T_amb, Q_fan_planned[r], forecasts["Solar"][0], 0.0] for r in rooms}
    observer._state["u_prev"] = u_current
    observer._save()

    # 6. Log
    logger.log(
        timestamp, measurements, forecasts, optimal_action, written,
        observer.x_state, solve_time, mip_gap,
        "dry_run_optimal" if dry_run else solver_status,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

DRY_RUN_DEFAULT = True


def _seconds_to_next_15min() -> float:
    now = datetime.now(timezone.utc)
    seconds_into_block = (now.minute % 15) * 60 + now.second + now.microsecond / 1e6
    return math.ceil(900 - seconds_into_block)


def main(dry_run: bool, immediate: bool = False, mode_end: datetime | None = None) -> None:
    log.info("Loading MPC config...")
    config = load_mpc_config()
    rc_models = config["rc_models"]

    log.info("Building parametric Gurobi model (once)...")
    m, params, vars_dict = build_parametric_mpc(HORIZON_STEPS, config)

    log.info("Initializing state reader and observer...")
    state_reader = StateReader(meter_config_file="meters_demo.yaml")
    observer = RCObserver(rc_models=rc_models, state_file=STATE_FILE)
    if observer.needs_warmup:
        observer.warmup_from_history(state_reader)

    forecast_provider = ForecastProvider()
    logger = StepLogger(log_file=LOG_FILE)
    fallback_state = {"consecutive_failures": 0, "last_plan": None, "predicted_next": None}

    if mode_end is not None:
        log.info(f"mode_end={mode_end.isoformat()} -- comfort-schedule tightening will be "
                  f"suppressed at/after this time (handing off to another controller).")

    if dry_run:
        log.info("=== DRY RUN — no PLC connection, no writes ===")
        run_one_step(
            m, params, vars_dict,
            state_reader, observer, forecast_provider,
            control_writer=None, logger=logger, rc_models=rc_models,
            fallback_state=fallback_state, dry_run=True, mode_end=mode_end,
        )
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
            run_one_step(
                m, params, vars_dict,
                state_reader, observer, forecast_provider,
                control_writer, logger, rc_models,
                fallback_state, mode_end=mode_end,
            )
        except Exception as e:
            log.exception(f"Unhandled error in control step: {e}")

        wait = _seconds_to_next_15min()
        log.info(f"Next step in {wait}s.")
        time.sleep(wait)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", dest="dry_run", action="store_true", default=None,
        help="One solve, no PLC connection, no writes -- saves plan plot to figures/ and exits."
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
