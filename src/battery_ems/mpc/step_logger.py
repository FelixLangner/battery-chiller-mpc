import json
from pathlib import Path
from datetime import datetime


class StepLogger:
    """
    Appends one JSON record per MPC step to a .jsonl file.

    Each line is a self-contained record: load the full log with
        pd.read_json("data/mpc_logs/mpc_steps.jsonl", lines=True)
    """

    def __init__(self, log_file: Path):
        self.log_file = log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        timestamp: datetime,
        measurements: dict,
        forecasts: dict,
        optimal_action: dict | None,
        written: dict | None,
        x_state: dict,
        solve_time_s: float,
        mip_gap: float | None,
        solver_status: str,
    ) -> None:
        record = {
            "timestamp": timestamp.isoformat(),
            "T_amb": measurements.get("T_amb"),
            "T_sup": measurements.get("T_sup"),
            "chiller_running": measurements.get("chiller_running"),
            "room_temps": measurements.get("room_temps"),
            "Q_fan_measured": measurements.get("Q_fan_measured"),
            "forecast_T_amb": forecasts.get("T_amb"),
            "forecast_Solar": forecasts.get("Solar"),
            "forecast_tariffs": forecasts.get("tariffs"),
            "forecast_T_min": forecasts.get("T_min"),
            "forecast_T_max": forecasts.get("T_max"),
            "forecast_T_sup_min": forecasts.get("T_sup_min"),
            "x_state": x_state,
            "optimal_action": optimal_action,
            "written": written,
            "solve_time_s": solve_time_s,
            "mip_gap": mip_gap,
            "solver_status": solver_status,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
