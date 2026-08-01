from unittest.mock import MagicMock

from battery_ems.mpc.control_writer import (
    CHILLER_OFF_SETPOINT,
    CHILLER_ON_SETPOINT,
    ControlWriter,
)


def _action(chiller_on: bool, **fan_on: bool) -> dict:
    return {"Chiller_Command": int(chiller_on), "Fan_Commands": {r: int(v) for r, v in fan_on.items()}}


def test_write_sends_chiller_three_times_first_step():
    plc = MagicMock()
    writer = ControlWriter(plc)

    writer.write(_action(True, room_1=True, room_2=False))

    assert plc.set_supply_temp.call_count == 3
    plc.set_supply_temp.assert_called_with(CHILLER_ON_SETPOINT, building="demo")
    assert plc.set_fan_speed.call_count == 6  # 2 rooms x 3 attempts
    plc.set_fan_speed.assert_any_call("normal", room="R1", building="demo")
    plc.set_fan_speed.assert_any_call("off", room="R2", building="demo")


def test_write_always_resends_chiller_even_when_unchanged():
    plc = MagicMock()
    writer = ControlWriter(plc)
    action = _action(True, room_1=True)

    writer.write(action)
    writer.write(action)  # identical decision next step -- chiller still resent

    assert plc.set_supply_temp.call_count == 6  # 3 + 3, never deduplicated
    plc.set_supply_temp.assert_called_with(CHILLER_ON_SETPOINT, building="demo")


def test_write_skips_unchanged_fan_channels_across_steps():
    plc = MagicMock()
    writer = ControlWriter(plc)
    action = _action(True, room_1=True, room_2=False)

    writer.write(action)
    writer.write(action)  # identical decision next step -- fans not re-sent

    assert plc.set_fan_speed.call_count == 6  # only from the first write
    assert plc.set_supply_temp.call_count == 6  # chiller resent both steps (3+3)


def test_write_resends_only_the_fan_channel_that_changed():
    plc = MagicMock()
    writer = ControlWriter(plc)

    writer.write(_action(True, room_1=True, room_2=False))
    writer.write(_action(True, room_1=False, room_2=False))  # only room_1 flips

    assert plc.set_fan_speed.call_count == 9  # 6 initial + 3 for room_1's flip
    plc.set_fan_speed.assert_any_call("off", room="R1", building="demo")


def test_write_resends_chiller_when_command_flips():
    plc = MagicMock()
    writer = ControlWriter(plc)

    writer.write(_action(True, room_1=True))
    writer.write(_action(False, room_1=True))

    assert plc.set_supply_temp.call_count == 6  # 3 + 3
    plc.set_supply_temp.assert_any_call(CHILLER_ON_SETPOINT, building="demo")
    plc.set_supply_temp.assert_any_call(CHILLER_OFF_SETPOINT, building="demo")


def test_write_hard_fallback_always_sends_chiller_but_dedups_room_setpoint():
    plc = MagicMock()
    writer = ControlWriter(plc)

    writer.write_hard_fallback()
    writer.write_hard_fallback()  # repeated fallback, nothing changed

    assert plc.set_supply_temp.call_count == 2  # chiller never deduplicated
    plc.set_supply_temp.assert_called_with(10, building="demo")
    plc.reset_room_setpoints.assert_called_once_with(setpoint=22)  # room setpoint still deduped


def test_write_hard_fallback_chiller_write_is_independent_of_normal_write():
    plc = MagicMock()
    writer = ControlWriter(plc)

    writer.write(_action(True, room_1=True))  # 3x supply=CHILLER_ON_SETPOINT (8)
    writer.write_hard_fallback()  # fallback wants 10 -- always sent, no dedup either way

    assert plc.set_supply_temp.call_count == 4  # 3 (normal write) + 1 (fallback)
    plc.set_supply_temp.assert_any_call(CHILLER_ON_SETPOINT, building="demo")
    plc.set_supply_temp.assert_any_call(10, building="demo")
