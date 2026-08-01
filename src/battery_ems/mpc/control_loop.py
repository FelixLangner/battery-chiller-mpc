"""The per-control-step pipeline: read sensors, update the Kalman observer,
fetch forecasts, solve the MPC, apply the decision, log."""
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from gurobipy import GRB

from battery_ems.controllers.gurobipy_mpc import ParametricMPC
from battery_ems.mpc.fallback_state import FallbackState
from battery_ems.mpc.forecast_provider import ForecastProvider
from battery_ems.mpc.plotting import plot_mpc_results
from battery_ems.mpc.prediction_writer import write_predictions
from battery_ems.mpc.pv_load_forecast import PVLoadForecastProvider
from battery_ems.mpc.rc_observer import RCObserver
from battery_ems.mpc.sensor_gaps import fill_sensor_gaps
from battery_ems.mpc.state_reader import StateReader
from battery_ems.mpc.step_logger import StepLogger

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent.parent  # mpc/ -> battery_ems/ -> src/ -> project root
FIGURES_DIR = ROOT / "figures"

_GRB_STATUS_NAMES = {
    GRB.OPTIMAL: "optimal",
    GRB.INFEASIBLE: "infeasible",
    GRB.UNBOUNDED: "unbounded",
    GRB.TIME_LIMIT: "time_limit",
}


@dataclass
class MPCControlLoop:
    mpc: ParametricMPC
    state_reader: StateReader
    observer: RCObserver
    forecast_provider: ForecastProvider
    pv_load_forecast_provider: PVLoadForecastProvider
    logger: StepLogger
    rc_models: dict
    battery_capacity_kwh: float
    fallback: FallbackState = field(default_factory=FallbackState)

    @property
    def rooms(self) -> list[str]:
        return list(self.rc_models.keys())

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_one_step(self, control_writer, dry_run: bool = False,
                      mode_end: datetime | None = None) -> None:
        timestamp = datetime.now(timezone.utc)
        rooms = self.rooms

        # 1. Read sensors & fill NaN gaps
        measurements = self._read_sensors()
        if measurements is None:
            return
        T_amb = measurements["T_amb"]

        # 2. Update Kalman observer
        initial_states = self._update_observer(measurements)

        # 3. Forecasts: weather/price/comfort + PV/load
        forecasts = self._get_forecasts(T_amb, timestamp, mode_end)

        # 4. Solve MPC
        solved = self._solve(initial_states, forecasts, timestamp, measurements)
        if solved is None:
            return
        optimal_action, solve_time, mip_gap, solver_status = solved

        if optimal_action is None:
            self._handle_infeasible(control_writer, dry_run, timestamp, measurements, forecasts,
                                     solve_time, mip_gap)
            return

        self.fallback.consecutive_failures = 0
        self.fallback.last_plan = optimal_action

        vars_dict = self.mpc.vars_dict
        block_size, holdover_blocks = self.mpc.block_size, self.mpc.holdover_blocks
        self.fallback.predicted_next = [
            {
                "T_sup": vars_dict["T_sup"][k * block_size].X,
                "T_amb": forecasts["T_amb"][k * block_size],
                "Q_fan": {r: vars_dict["Q_fan"][r][k * block_size - 1].X for r in rooms},
                "SOC": vars_dict["SOC"][k * block_size].X,
            }
            for k in range(1, holdover_blocks + 1)
        ]

        batt_kw = optimal_action["Battery_Power_kW"]
        log.info(
            f"Decision: chiller={'ON' if optimal_action['Chiller_Command'] else 'OFF'}, "
            f"fans={optimal_action['Fan_Commands']}, "
            f"battery={abs(batt_kw):.2f}kW {'discharge' if batt_kw >= 0 else 'charge'}"
        )

        # 5. Write to PLC (or save plan plot in dry-run), persist observer state
        written = self._apply_decision(forecasts, optimal_action, T_amb, timestamp, control_writer, dry_run)

        # 6. Log
        self.logger.log(
            timestamp, measurements, forecasts, optimal_action, written,
            self.observer.x_state, solve_time, mip_gap,
            "dry_run_optimal" if dry_run else solver_status,
        )

    # ------------------------------------------------------------------
    # Step internals
    # ------------------------------------------------------------------

    def _read_sensors(self) -> dict | None:
        """Sensor read + NaN gap-fill. Returns None (already logged) if the read itself fails."""
        try:
            measurements = self.state_reader.read()
        except Exception as e:  # noqa: BLE001 -- any sensor-read failure must not crash the loop
            log.error(f"Sensor read failed: {e}. Skipping step.")
            return None
        return fill_sensor_gaps(measurements, self.observer, self.rc_models,
                                 self.battery_capacity_kwh, self.fallback)

    def _update_observer(self, measurements: dict) -> dict:
        """Kalman update from the previous step's control input; returns the
        initial-state dict mpc.step() needs for this solve."""
        observer, rooms = self.observer, self.rooms
        T_amb = measurements["T_amb"]
        T_sup = measurements["T_sup"]
        T_lift_historical = measurements["T_amb_5min_ago"] - measurements["T_sup_5min_ago"]

        u_prev_stored = observer.get_u_prev(rooms, T_amb_fallback=T_amb)
        u_prev = {
            r: [u_prev_stored[r][0], measurements["Q_fan_measured"][r], u_prev_stored[r][2]]
            for r in rooms
        }
        observer.update(
            y_measured=measurements["room_temps_history"],
            u_prev=u_prev,
            T_amb=T_amb,
            T_sup=T_sup,
        )
        return {
            "T_sup_current": T_sup,
            "Temp_Lift_historical": T_lift_historical,
            "x_state_current": observer.x_state,
            "Chiller_On_prev": measurements["chiller_running"],
            "SOC_current": measurements["battery_soc_kwh"],
        }

    def _get_forecasts(self, T_amb: float, timestamp: datetime, mode_end: datetime | None) -> dict:
        """Weather/price/comfort forecast merged with the PV/load forecast."""
        horizon_steps = self.mpc.horizon_steps
        forecasts = self.forecast_provider.get(horizon_steps, T_amb_current=T_amb, rooms=self.rooms,
                                                mode_end=mode_end)
        pv_forecast, load_forecast = self.pv_load_forecast_provider.get(horizon_steps, now=timestamp)
        forecasts["PV_forecast"] = pv_forecast
        forecasts["Load_forecast"] = load_forecast
        return forecasts

    def _solve(self, initial_states: dict, forecasts: dict, timestamp: datetime, measurements: dict):
        """Runs mpc.step() and times/classifies the result. Returns
        (optimal_action, solve_time, mip_gap, solver_status), or None if the solve itself
        raised (already logged and step-logged as 'exception')."""
        mpc = self.mpc
        t0 = time.time()
        try:
            optimal_action = mpc.step(initial_states, forecasts)
            solve_time = time.time() - t0
            mip_gap = mpc.m.MIPGap if mpc.m.SolCount > 0 else None
            solver_status = _GRB_STATUS_NAMES.get(mpc.m.Status, f"status_{mpc.m.Status}")
        except Exception as e:  # noqa: BLE001 -- any solve failure must not crash the loop
            log.error(f"MPC solve exception: {e}")
            self.logger.log(timestamp, measurements, forecasts, None, None,
                             self.observer.x_state, time.time() - t0, None, "exception")
            return None

        gap_str = f"{mip_gap:.4f}" if mip_gap is not None else "N/A"
        log.info(f"Solve: {solver_status}, gap={gap_str}, t={solve_time:.1f}s")
        return optimal_action, solve_time, mip_gap, solver_status

    def _apply_decision(self, forecasts: dict, optimal_action: dict, T_amb: float, timestamp: datetime,
                         control_writer, dry_run: bool) -> dict | None:
        """Best-effort prediction write, then either a PLC write or a dry-run plan plot,
        then persists the observer's u_prev for next step. Returns what was actually
        written (None in dry-run or on a PLC write failure)."""
        mpc, rooms, observer = self.mpc, self.rooms, self.observer
        vars_dict = mpc.vars_dict
        if not dry_run:
            try:
                write_predictions(timestamp, forecasts, vars_dict, rooms, mpc.horizon_steps)
            except Exception as e:  # noqa: BLE001 -- best-effort write, must not abort the step
                log.warning(f"Writing predictions failed: {e}")

        if dry_run:
            log.info("[DRY RUN] Skipping PLC write.")
            written = None
            self._save_plan_plot(forecasts, timestamp)
        else:
            try:
                written = control_writer.write(optimal_action)
            except Exception as e:  # noqa: BLE001 -- PLC write failure must not crash the loop
                log.error(f"PLC write failed: {e}")
                written = None

        Q_fan_planned = {r: vars_dict["Q_fan"][r][0].X for r in rooms}
        u_current = {r: [T_amb, Q_fan_planned[r], forecasts["Solar"][0]] for r in rooms}
        observer._state["u_prev"] = u_current
        observer._save()

        return written

    def _handle_infeasible(self, control_writer, dry_run: bool, timestamp, measurements, forecasts,
                            solve_time, mip_gap) -> None:
        fallback = self.fallback
        fallback.consecutive_failures += 1
        n = fallback.consecutive_failures
        holdover_blocks = self.mpc.holdover_blocks

        if n <= holdover_blocks and fallback.last_plan is not None:
            plan = fallback.last_plan
            holdover = {
                "Chiller_Command": plan["Holdover_Chiller_Commands"][n - 1],
                "Fan_Commands":    plan["Holdover_Fan_Commands"][n - 1],
                "Battery_Power_kW": plan["Holdover_Battery_Power_kW"][n - 1],
            }
            log.warning(
                f"MPC infeasible (#{n}/{holdover_blocks}) — holdover block k={n} of last plan: "
                f"chiller={'ON' if holdover['Chiller_Command'] else 'OFF'}, "
                f"fans={holdover['Fan_Commands']}, battery={holdover['Battery_Power_kW']:.2f}kW"
            )
            written = None
            if not dry_run:
                try:
                    written = control_writer.write(holdover)
                except Exception as e:  # noqa: BLE001 -- PLC write failure must not crash the loop
                    log.error(f"PLC holdover write failed: {e}")
            self.logger.log(timestamp, measurements, forecasts, holdover, written,
                             self.observer.x_state, solve_time, mip_gap, "infeasible_holdover")
        else:
            log.error(f"MPC infeasible (#{n}) — hard fallback: supply=10°C, rooms=22°C, battery idle.")
            written = None
            if not dry_run:
                try:
                    written = control_writer.write_hard_fallback()
                except Exception as e:  # noqa: BLE001 -- PLC write failure must not crash the loop
                    log.error(f"PLC hard-fallback write failed: {e}")
            self.logger.log(timestamp, measurements, forecasts, None, written,
                             self.observer.x_state, solve_time, mip_gap, "infeasible_hard_fallback")

    def _save_plan_plot(self, forecasts: dict, timestamp: datetime) -> None:
        FIGURES_DIR.mkdir(exist_ok=True)
        out = FIGURES_DIR / f"dry_run_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
        plot_mpc_results(self.mpc, forecasts, save_path=out)
        log.info(f"[DRY RUN] Plan plots saved: {out.stem}_cooling{out.suffix} / {out.stem}_battery{out.suffix}")
