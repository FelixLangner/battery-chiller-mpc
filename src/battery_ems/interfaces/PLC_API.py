"""
Synthetic drop-in replacement for the private repo's real vendor PLC client.
Same 3-method contract control_writer.py actually calls
(`set_supply_temp`, `set_fan_speed`, `reset_room_setpoints`), so
control_writer.py needs zero changes.

There's no real hardware here, so instead of a network client this binds to
a live `SyntheticBuilding` instance (see emulators/synthetic_building/plant.py)
via a small module-level bind, set once by whichever entrypoint owns the
simulated clock (run_synthetic_demo.py) before starting the control loop --
mirrors how, in the private repo, PLC_API and EnergyDataInterface are two
independent connections that happen to reflect the same physical building;
here they're two independent objects that happen to reflect the same
SyntheticBuilding instance.

For `run_mpc.py --dry-run`, PLC_API is imported but never instantiated or
bound -- the dry-run path never calls it.
"""
import logging

log = logging.getLogger(__name__)

_bound_building = None


def bind_building(building) -> None:
    """Call once, before starting a live control loop, to connect PLC_API
    writes to a running SyntheticBuilding instance."""
    global _bound_building
    _bound_building = building


class PLC_API:
    """Synthetic stand-in for the real vendor PLC client."""

    def __init__(self, buildings: list[str] | None = None):
        self.buildings = buildings or []
        if _bound_building is None:
            log.warning("PLC_API constructed with no bound SyntheticBuilding — "
                        "writes will be no-ops until bind_building() is called.")

    def set_supply_temp(self, setpoint: float, building: str = "demo") -> None:
        if _bound_building is not None:
            _bound_building.set_supply_setpoint(float(setpoint))

    def set_fan_speed(self, speed: str, room: str, building: str = "demo") -> None:
        if _bound_building is not None:
            mpc_room = f"room_{room[1:]}"  # "R1" -> "room_1"
            _bound_building.set_fan_speed(mpc_room, speed)

    def reset_room_setpoints(self, setpoint: float) -> None:
        """
        Hard-fallback path only (after repeated infeasible solves). The real
        vendor API writes a room-thermostat setpoint here, a control channel
        this simplified synthetic plant doesn't separately model (it only
        simulates chiller supply-setpoint + per-room fan on/off) -- treated
        as a no-op, logged for visibility. This only affects the rare
        infeasible-solve fallback path, not normal operation.
        """
        log.warning(f"reset_room_setpoints({setpoint}) — no-op in the synthetic plant "
                    f"(no separate room-thermostat channel modeled).")
