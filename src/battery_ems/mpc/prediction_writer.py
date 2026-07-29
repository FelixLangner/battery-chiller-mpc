"""
No-op stand-in for the private repo's prediction_writer.py (which pushes
each solve's full predicted trajectory to a second, independent InfluxDB
instance for later prediction-vs-reality analysis). That's a debugging/
analysis feature, not control-critical -- run_mpc.py already calls it inside
a best-effort try/except and skips it entirely in dry-run -- so for the demo
it's stubbed to a no-op rather than building a second synthetic store type.
Same function signature as the real module, so run_mpc.py needs no changes.
"""
import logging
from datetime import datetime

log = logging.getLogger(__name__)


def write_predictions(timestamp: datetime, forecasts: dict, vars_dict: dict,
                       rooms: list[str], horizon_steps: int) -> None:
    log.debug("write_predictions() is a no-op in the synthetic demo (no second store).")
