from datetime import datetime, timedelta, timezone

from battery_ems.mpc.forecast_provider import ForecastProvider

# The demo's forecast_provider.py uses a uniform comfort schedule (no
# per-room override), unlike the private repo's ROOM_T_MAX_OVERRIDE/
# ROOM_T_MIN_OVERRIDE -- so unlike the original tests this ports, every
# room is expected to behave identically.
ROOMS = ["room_1", "room_4", "room_5"]
STEP_MINUTES = 5


def _step_time(now: datetime, i: int) -> datetime:
    return now + timedelta(minutes=i * STEP_MINUTES)


# --- _comfort_schedule: baseline behaviour (no mode_end) --------------------

def test_comfort_schedule_office_hours_tight_no_mode_end():
    # 09:00 Berlin (CEST, UTC+2) -> 07:00 UTC, well inside office hours
    now = datetime(2026, 7, 22, 7, 0, tzinfo=timezone.utc)
    T_min, T_max = ForecastProvider._comfort_schedule(1, now, ROOMS)
    for r in ROOMS:
        assert T_min[r] == [22.0]
        assert T_max[r] == [24.0]


def test_comfort_schedule_night_relaxed_no_mode_end():
    # 22:00 Berlin -> 20:00 UTC, well outside office hours
    now = datetime(2026, 7, 22, 20, 0, tzinfo=timezone.utc)
    T_min, T_max = ForecastProvider._comfort_schedule(1, now, ROOMS)
    for r in ROOMS:
        assert T_min[r] == [20.0]
        assert T_max[r] == [28.0]


# --- _comfort_schedule: mode_end suppresses tightening past a handoff ------

def test_comfort_schedule_mode_end_suppresses_future_office_hours_tightening():
    # now = 20:00 UTC on day 1 (night). Horizon runs long enough to reach the
    # NEXT day's office-hours window (07:00-18:00 Berlin = 05:00-16:00 UTC in
    # summer). mode_end is set to right at that next office-hours start, so
    # the MPC should NOT see the tightening at all -- every step stays relaxed.
    now = datetime(2026, 7, 22, 20, 0, tzinfo=timezone.utc)
    next_office_start = datetime(2026, 7, 23, 5, 0, tzinfo=timezone.utc)  # 07:00 Berlin
    horizon_steps = int((timedelta(hours=10)) / timedelta(minutes=STEP_MINUTES))  # well past next_office_start

    T_min, T_max = ForecastProvider._comfort_schedule(
        horizon_steps, now, ROOMS, mode_end=next_office_start
    )

    for i in range(horizon_steps):
        t = _step_time(now, i)
        if t >= next_office_start:
            assert T_max["room_1"][i] == 28.0, f"step {i} ({t}) should stay relaxed past mode_end"
            assert T_min["room_1"][i] == 20.0


def test_comfort_schedule_mode_end_does_not_affect_steps_before_handoff():
    # now is already inside office hours; mode_end is later in the future
    # (past the whole horizon) -- behaviour must be identical to no mode_end.
    now = datetime(2026, 7, 22, 7, 0, tzinfo=timezone.utc)  # 09:00 Berlin, office hours
    far_future_mode_end = now + timedelta(days=30)
    horizon_steps = 5

    T_min_no_cutoff, T_max_no_cutoff = ForecastProvider._comfort_schedule(horizon_steps, now, ROOMS)
    T_min_far_cutoff, T_max_far_cutoff = ForecastProvider._comfort_schedule(
        horizon_steps, now, ROOMS, mode_end=far_future_mode_end
    )

    assert T_min_no_cutoff == T_min_far_cutoff
    assert T_max_no_cutoff == T_max_far_cutoff


def test_comfort_schedule_mode_end_exactly_at_step_time_counts_as_past_handoff():
    # mode_end == a step's own timestamp should already suppress that step
    # (>=, not >) -- the handoff is effectively instantaneous at that instant.
    now = datetime(2026, 7, 22, 7, 0, tzinfo=timezone.utc)  # office hours
    mode_end = now  # handoff is "now"
    T_min, T_max = ForecastProvider._comfort_schedule(1, now, ROOMS, mode_end=mode_end)
    assert T_max["room_1"] == [28.0]


def test_comfort_schedule_mode_end_none_is_a_no_op():
    now = datetime(2026, 7, 22, 7, 0, tzinfo=timezone.utc)
    T_min_a, T_max_a = ForecastProvider._comfort_schedule(10, now, ROOMS, mode_end=None)
    T_min_b, T_max_b = ForecastProvider._comfort_schedule(10, now, ROOMS)
    assert T_min_a == T_min_b
    assert T_max_a == T_max_b
