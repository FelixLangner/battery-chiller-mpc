"""
Fast-forward closed-loop demo: the real run_mpc.py control-step logic
(unmodified, imported directly), driven against a live SyntheticBuilding for
the full MPC loop.
Usage:
    python -m scripts.run_synthetic_demo [--cycles N]

Requires the store to already have some history (run
scripts/seed_synthetic_history.py first) so the observer doesn't need a
cold start, though it works from a cold start too (see run_mpc.py).

Ambient temperature runs in a heat-wave band by default (see
T_AMB_OFFSET/SyntheticBuilding's T_amb_offset) and the live DWD forecast
fetch is replaced with the same hot diurnal shape, so the chiller actually
has to run to hold the 22-24°C daytime comfort band -- the unshifted
18-30°C band rarely forces it (fans alone can usually cope, and a small,
cheaply-penalized comfort overshoot can beat paying to run the chiller at
+10°C offset), so a plain run would otherwise show chiller=OFF throughout.
"""
import argparse
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")  # set before any transitive `import matplotlib.pyplot` (e.g. via control_loop -> plotting)
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import battery_ems.interfaces.influx as influx_module
import battery_ems.interfaces.plc_api as plc_module
import battery_ems.mpc.forecast_provider as forecast_provider_module
import battery_ems.mpc.state_reader as state_reader_module
from battery_ems.controllers.gurobipy_mpc import ParametricMPC
from battery_ems.emulators.synthetic_building.plant import (
    SyntheticBuilding,
)
from battery_ems.emulators.synthetic_building.store import TimeSeriesStore
from battery_ems.interfaces.influx import EnergyDataInterface
from battery_ems.interfaces.plc_api import PLC_API
from battery_ems.mpc.control_loop import MPCControlLoop
from battery_ems.mpc.control_writer import ControlWriter
from battery_ems.mpc.forecast_provider import ForecastProvider
from battery_ems.mpc.pv_load_forecast import PVLoadForecastProvider
from battery_ems.mpc.rc_observer import RCObserver
from battery_ems.mpc.run_mpc import (
    HORIZON_STEPS,
    LOG_FILE,
    STATE_FILE,
    load_mpc_config,
    setup_logging,
)
from battery_ems.mpc.state_reader import StateReader
from battery_ems.mpc.step_logger import StepLogger

STORE_PATH = REPO / "data" / "synthetic_store.parquet"

T_AMB_OFFSET = 18.0  # -> 36-48°C diurnal band, see SyntheticBuilding.T_amb_offset
_BERLIN = ZoneInfo("Europe/Berlin")


def _next_office_hours_start(after: datetime) -> datetime:
    """Next Berlin-local 09:00 at/after `after` -- mid office-hours, so the
    tight 22-24°C comfort band (forecast_provider.py's _comfort_schedule) is
    active from the first cycle. Continuing from wherever the store's clock
    happened to stop can land in the relaxed 20-28°C night/weekend band,
    where hot ambient legitimately doesn't need cooling -- that's what an
    early version of this demo actually showed (chiller stayed OFF at
    ~21:00 Berlin)."""
    local = after.astimezone(_BERLIN)
    candidate = local.replace(hour=9, minute=0, second=0, microsecond=0)
    if candidate <= local:
        candidate += timedelta(days=1)
    return candidate.astimezone(timezone.utc)


def _plot_closed_loop_trajectory(sim_times, room_temps: dict, T_min_hist, T_max_hist,
                                  chiller_hist, soc_hist, battery_capacity_kwh: float,
                                  out_path: Path) -> None:
    """What was actually executed, cycle by cycle -- not a single solve's plan
    (see mpc/plotting.py::plot_mpc_results for that). This is the closed-loop
    view: does the real room temperature actually stay in its comfort band,
    and does the chiller actually engage when it needs to."""
    t_hours = [(t - sim_times[0]).total_seconds() / 3600.0 for t in sim_times]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    ax1.set_title("Closed-loop room temperature vs. comfort band", fontweight="bold")
    for room, temps in room_temps.items():
        ax1.plot(t_hours, temps, linewidth=1.8, alpha=0.85, label=room)
    ax1.plot(t_hours, T_min_hist, "k--", linewidth=1, alpha=0.6, label="comfort band")
    ax1.plot(t_hours, T_max_hist, "k--", linewidth=1, alpha=0.6)
    ax1.set_ylabel("Room temp (°C)", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", ncol=len(room_temps) + 1, fontsize="small")

    ax1_twin = ax1.twinx()
    ax1_twin.step(t_hours, [1 if c else 0 for c in chiller_hist], where="post",
                  color="#D95319", linewidth=2, alpha=0.7, label="Chiller")
    ax1_twin.set_ylabel("Chiller (ON/OFF)", color="#D95319", fontweight="bold")
    ax1_twin.set_ylim(-0.1, 1.1)

    ax2.set_title("Battery state of charge", fontweight="bold")
    ax2.plot(t_hours, [s / battery_capacity_kwh for s in soc_hist], color="#2a78d6", linewidth=2)
    ax2.set_ylabel("SOC (0-1)", fontweight="bold")
    ax2.set_xlabel("Simulated time (hours since run start)", fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _forecast_fetch(horizon_steps: int, now: datetime, T_amb_fallback: float):
    """Drop-in replacement for ForecastProvider._fetch that skips the live DWD
    call and instead projects the same hot diurnal shape SyntheticBuilding's
    ground truth uses (see plant.py::_diurnal_weather) -- keeps the MPC's
    forecast consistent with the hot ground truth across the whole horizon,
    not just the first 15 minutes (which _slice() would anchor to
    T_amb_fallback regardless, but everything past that would otherwise
    revert to whatever DWD's real current weather happens to be)."""
    Ta, Solar = [], []
    for i in range(horizon_steps):
        hour = (now + timedelta(minutes=5 * i)).hour
        Ta.append(24.0 + T_AMB_OFFSET + 6.0 * math.sin(2 * math.pi * (hour - 8) / 24))
        Solar.append(max(0.0, 0.7 * math.sin(math.pi * (hour - 6) / 13)) if 6 <= hour <= 19 else 0.0)
    return Ta, Solar


def main(cycles: int) -> None:
    setup_logging()

    store = TimeSeriesStore(STORE_PATH)
    if len(store) == 0:
        print("WARNING: store is empty -- run scripts/seed_synthetic_history.py "
              "first for a realistic demo. Continuing with a cold start.")
        start_time = datetime.now(timezone.utc)
    else:
        start_time = store._df.index[-1].to_pydatetime()

    start_time = _next_office_hours_start(start_time)
    building = SyntheticBuilding(store, start_time=start_time, seed=7, T_amb_offset=T_AMB_OFFSET)
    plc_module.bind_building(building)

    state_reader_module.set_clock(lambda: building.time)
    influx_module.set_clock(lambda: building.time)
    forecast_provider_module.set_clock(lambda: building.time)

    print("Loading MPC config...")
    config = load_mpc_config()
    rc_models = config["rc_models"]
    battery_capacity_kwh = config["battery_physics"]["capacity_kwh"]

    print("Building parametric joint MPC model (once)...")
    mpc = ParametricMPC(HORIZON_STEPS, config)

    shared_db = EnergyDataInterface(store=store)
    state_reader = StateReader(meter_config_file="meters_demo.yaml", db=shared_db)
    observer = RCObserver(rc_models=rc_models, state_file=STATE_FILE)
    if observer.needs_warmup:
        observer.warmup_from_history(state_reader)

    forecast_provider = ForecastProvider()
    band_lo, band_hi = 24.0 + T_AMB_OFFSET - 6.0, 24.0 + T_AMB_OFFSET + 6.0
    print(f"=== T_amb_offset=+{T_AMB_OFFSET}°C ({band_lo:.0f}-{band_hi:.0f}°C band) -- "
          f"forecast uses the same hot shape instead of live DWD, to force chiller-ON. ===")
    forecast_provider._fetch = _forecast_fetch

    loop = MPCControlLoop(
        mpc=mpc,
        state_reader=state_reader,
        observer=observer,
        forecast_provider=forecast_provider,
        pv_load_forecast_provider=PVLoadForecastProvider(),
        logger=StepLogger(log_file=LOG_FILE),
        rc_models=rc_models,
        battery_capacity_kwh=battery_capacity_kwh,
    )

    plc_api = PLC_API(buildings=["demo"])
    control_writer = ControlWriter(plc_api=plc_api, building="demo")

    print(f"Starting fast-forward closed loop: {cycles} cycles of 15 simulated minutes each "
          f"(building clock starts at {building.time.isoformat()}).")

    rooms = list(rc_models.keys())
    sim_times, room_temp_history = [], {r: [] for r in rooms}
    T_min_history, T_max_history, chiller_history, soc_history = [], [], [], []
    for cycle in range(cycles):
        # Advance physics 15 sim-minutes (3 x 5-min steps) under whatever
        # was last commanded, THEN read/solve/write for the next interval --
        # mirrors real causality (a decision only takes effect going
        # forward, not retroactively).
        for _ in range(mpc.block_size):
            building.step()

        print(f"\n--- Cycle {cycle + 1}/{cycles} | sim time {building.time.isoformat()} "
              f"| room_1={building.rooms['room_1'].T_room:.2f}°C | "
              f"chiller={'ON' if building.chiller_on else 'OFF'} | "
              f"battery_soc={building.battery.soc_kwh:.2f}/{battery_capacity_kwh:.1f}kWh ---")

        sim_times.append(building.time)
        for r in rooms:
            room_temp_history[r].append(building.rooms[r].T_room)
        T_min_now, T_max_now = ForecastProvider._comfort_schedule(1, building.time, ["room_1"])
        T_min_history.append(T_min_now["room_1"][0])
        T_max_history.append(T_max_now["room_1"][0])
        chiller_history.append(building.chiller_on)
        soc_history.append(building.battery.soc_kwh)

        loop.run_one_step(control_writer, dry_run=False)

    store.save()
    print(f"\nDone. {cycles} cycles simulated, store saved to {STORE_PATH}.")
    print(f"room_1 temperature trajectory (°C): {[round(t, 2) for t in room_temp_history['room_1']]}")

    FIGURES_DIR = REPO / "figures"
    FIGURES_DIR.mkdir(exist_ok=True)
    plot_path = FIGURES_DIR / "synthetic_demo_trajectory.png"
    _plot_closed_loop_trajectory(sim_times, room_temp_history, T_min_history, T_max_history,
                                  chiller_history, soc_history, battery_capacity_kwh, plot_path)
    print(f"Closed-loop trajectory plot saved: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=8, help="Number of 15-min control cycles to run.")
    args = parser.parse_args()
    main(args.cycles)
