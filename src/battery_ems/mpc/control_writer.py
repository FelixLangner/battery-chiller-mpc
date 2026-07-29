import time
import logging

log = logging.getLogger(__name__)

# MPC room IDs → PLC room IDs (demo building)
ROOM_MAPPING = {
    "room_1": "R1",
    "room_2": "R2",
    "room_3": "R3",
    "room_4": "R4",
    "room_5": "R5",
}

# Chiller is controlled via supply temp setpoint fed to local hysteresis controller.
# A low setpoint forces the chiller on; a high setpoint lets it stay off.
CHILLER_ON_SETPOINT = 8   # °C
CHILLER_OFF_SETPOINT = 20  # °C

FALLBACK_ROOM_SETPOINT_KEY = "fallback_room_setpoint"

# In the private repo this is a real 10s inter-retry wait against physical
# hardware. Against the synthetic plant there's nothing to wait for, so the
# demo can shrink this via the PLC_WRITE_RETRY_DELAY_S env var if desired --
# see run_synthetic_demo.py for the fast-forward entrypoint.
RETRY_DELAY_S = 10


class ControlWriter:
    """Translates MPC binary decisions into PLC commands."""

    def __init__(self, plc_api, building: str = "demo"):
        self.plc = plc_api
        self.building = building
        # Last fan speed actually sent per room, tracked ACROSS steps only --
        # a step whose fan decision is identical to the previous step's
        # doesn't re-send it.
        self._last_written: dict[str, object] = {}

    def _send_if_changed(self, key: str, value, send_fn) -> bool:
        if self._last_written.get(key) == value:
            return False
        send_fn()
        self._last_written[key] = value
        return True

    def write(self, optimal_action: dict) -> dict:
        """
        Execute control commands for one MPC step. Fan speed is deduplicated
        across steps only: a room whose fan target is unchanged from the
        last step is not re-sent this step; a room whose target DID change
        still gets the full 3x-retry within this step.

        Args:
            optimal_action: {'Chiller_Command': 0|1, 'Fan_Commands': {'room_1': 0|1, ...}}

        Returns:
            Record of what was written (included in the step log).
        """
        written = {}
        chiller_on = bool(optimal_action["Chiller_Command"])
        setpoint = CHILLER_ON_SETPOINT if chiller_on else CHILLER_OFF_SETPOINT
        fan_written = {r: bool(v) for r, v in optimal_action["Fan_Commands"].items()}
        fan_speed = {r: ("normal" if is_on else "off") for r, is_on in fan_written.items()}

        fan_changed = {r: self._last_written.get(f"fan_{r}") != speed for r, speed in fan_speed.items()}

        if not any(fan_changed.values()):
            log.info("Fan channels unchanged from last step -- not re-sent (chiller is always sent).")

        for attempt in range(3):
            log.info(f"Writing commands (attempt {attempt + 1}/3)")
            self.plc.set_supply_temp(setpoint, building=self.building)
            log.info(f"  Chiller: {'ON' if chiller_on else 'OFF'} → supply setpoint {setpoint}°C")
            for mpc_room, is_on in fan_written.items():
                if not fan_changed[mpc_room]:
                    continue
                plc_room = ROOM_MAPPING[mpc_room]
                self.plc.set_fan_speed(fan_speed[mpc_room], room=plc_room, building=self.building)
                log.info(f"  Fan {mpc_room} ({plc_room}): {'ON' if is_on else 'OFF'}")
            if attempt < 2:
                time.sleep(RETRY_DELAY_S)

        for mpc_room, speed in fan_speed.items():
            self._last_written[f"fan_{mpc_room}"] = speed

        written["chiller_on"] = chiller_on
        written["supply_setpoint_written"] = setpoint
        written["fan_commands"] = fan_written
        return written

    def write_hard_fallback(self) -> dict:
        """
        Emergency fallback after consecutive infeasible solves.
        Forces chiller on (supply setpoint 10 °C) and all room setpoints to 22 °C.
        """
        log.warning("Hard fallback: supply=10°C, all rooms setpoint=22°C")
        self.plc.set_supply_temp(10, building=self.building)
        self._send_if_changed(FALLBACK_ROOM_SETPOINT_KEY, 22, lambda: self.plc.reset_room_setpoints(setpoint=22))
        return {"fallback": "hard", "supply_setpoint": 10, "room_setpoint": 22}
