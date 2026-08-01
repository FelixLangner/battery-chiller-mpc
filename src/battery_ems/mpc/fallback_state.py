from dataclasses import dataclass


@dataclass
class FallbackState:
    """Tracks consecutive solve failures and the last successful plan, so
    MPCControlLoop._handle_infeasible knows whether to coast on a holdover
    block or drop to the hard fallback, and fill_sensor_gaps knows what to
    backfill NaNs with."""
    consecutive_failures: int = 0
    last_plan: dict | None = None
    predicted_next: list[dict] | None = None
