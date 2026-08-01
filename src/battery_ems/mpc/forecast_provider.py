import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from battery_ems.mpc.forecasts import Forecasting

_BERLIN = ZoneInfo("Europe/Berlin")
_ROOT = Path(__file__).parent.parent.parent.parent  # mpc/ -> battery_ems/ -> src/ -> project root

log = logging.getLogger(__name__)


def _clock() -> datetime:
    return datetime.now(timezone.utc)


def set_clock(fn) -> None:
    """Override wall-clock time for demo testing, so a fast-forwarded
    SyntheticBuilding's simulated clock -- not real wall-clock time -- drives
    which comfort band (office-hours vs. night/weekend) and tariff period
    apply. Same pattern as state_reader.py/influx.py's set_clock(); without
    it, get()'s comfort/tariff scheduling stays pinned to whenever the script
    happened to actually run, never advancing with simulated time."""
    global _clock
    _clock = fn

STEP_MINUTES = 5
CACHE_TTL_MINUTES = 55  # DWD MOSMIX updates hourly; re-fetch just before the hour

# Real buy tariff -- unlike the rest of this repo, this one genuinely is real
# data: awattar.de's real German day-ahead wholesale price for 2026-06-19
# (a single real day picked as representative, not a multi-day average) plus
# real network-fee/tax/VAT schedule, hour-of-day indexed. Not building- or
# person-identifying (a public day-ahead market price), but a deliberate,
# explicit exception to this repo's usual synthetic-only data -- see
# README.md's "Synthetic data" section. Deployed hourly buy price for hour h
# is _REAL_TARIFF_BY_HOUR[h]; see the file's own "_meta" key for the
# wholesale/fee/VAT breakdown this was derived from.
_TARIFF_FILE = _ROOT / "data" / "real_tariff_2026-06-19.json"
with open(_TARIFF_FILE) as _f:
    _REAL_TARIFF_BY_HOUR = {int(h): v for h, v in json.load(_f).items() if h != "_meta"}

# Synthetic feed-in/sell price: always below buy (grid arbitrage isn't free
# money), with a midday dip (10:00-16:00) when PV oversupply would plausibly
# depress wholesale prices -- gives the battery MPC a genuine economic
# incentive shape (charge cheap/midday, discharge into the evening peak).
_SELL_EUR_KWH = 0.10
_SELL_MIDDAY_DIP_EUR_KWH = 0.04
_SELL_MIDDAY_HOURS = range(10, 16)


class ForecastProvider:
    """
    Returns forecast arrays for the full MPC horizon. Weather (T_amb, Solar)
    comes from the live, public DWD MOSMIX product (see forecasts.py/dwd.py).
    The buy tariff is real (see _REAL_TARIFF_BY_HOUR above); everything else
    (sell tariff, comfort schedule) is synthetic.
    """

    def __init__(self):
        self._cache: dict | None = None
        self._cache_fetched_at: datetime | None = None

    def get(self, horizon_steps: int, T_amb_current: float, rooms: list[str],
            mode_end: datetime | None = None) -> dict:
        now = _clock()
        Ta, Solar = self._fetch(horizon_steps, now, T_amb_current)
        T_min, T_max = self._comfort_schedule(horizon_steps, now, rooms, mode_end=mode_end)
        return {
            "T_amb": Ta,
            "Solar": Solar,
            "tariffs": self._tariff_schedule(horizon_steps, now),
            "sell_tariffs": self._sell_tariff_schedule(horizon_steps, now),
            "T_min": T_min,
            "T_max": T_max,
            "T_sup_min": [5.0] * horizon_steps,
        }

    # ------------------------------------------------------------------

    def _fetch(self, horizon_steps: int, now: datetime, T_amb_fallback: float):
        if self._cache is not None and self._cache_fetched_at is not None:
            age_min = (now - self._cache_fetched_at).total_seconds() / 60
            if age_min < CACHE_TTL_MINUTES:
                return self._slice(horizon_steps, T_amb_fallback)

        try:
            horizon_hours = math.ceil(horizon_steps * STEP_MINUTES / 60) + 1
            berlin_now = datetime.now(_BERLIN).replace(tzinfo=None)
            qs_series, Ta_series = Forecasting(berlin_now, STEP_MINUTES, horizon_hours)
            self._cache = {"Ta": Ta_series, "qs": qs_series}
            self._cache_fetched_at = now
            log.info(f"DWD forecast fetched: {len(Ta_series)} steps at {STEP_MINUTES}-min resolution.")
        except Exception as exc:  # noqa: BLE001 -- live DWD fetch failure must not crash the loop
            log.warning(f"DWD forecast failed ({exc}). Using {'cached' if self._cache else 'flat-persistence'} fallback.")

        return self._slice(horizon_steps, T_amb_fallback)

    def _slice(self, horizon_steps: int, T_amb_fallback: float):
        if self._cache is None:
            return [T_amb_fallback] * horizon_steps, [0.0] * horizon_steps

        Ta_raw = list(self._cache["Ta"].iloc[:horizon_steps])
        qs_raw = [v / 1000.0 for v in self._cache["qs"].iloc[:horizon_steps]]  # W/m² → kW/m²

        if len(Ta_raw) < horizon_steps:
            pad = horizon_steps - len(Ta_raw)
            Ta_raw += [Ta_raw[-1] if Ta_raw else T_amb_fallback] * pad
            qs_raw += [qs_raw[-1] if qs_raw else 0.0] * pad

        # Anchor the whole first 15-min control block to the just-measured
        # T_amb rather than DWD's hourly "current hour" value.
        FIRST_BLOCK_STEPS = 3
        for i in range(min(FIRST_BLOCK_STEPS, len(Ta_raw))):
            Ta_raw[i] = T_amb_fallback

        return Ta_raw, qs_raw

    @staticmethod
    def _comfort_schedule(horizon_steps: int, now: datetime, rooms: list[str],
                          mode_end: datetime | None = None) -> tuple:
        """
        Office hours (7:00-18:00 Berlin local time): tight comfort band 22-24C.
        Night / weekend: relaxed to 20-28C so the MPC can pre-cool cheaply.
        """
        T_min = {r: [] for r in rooms}
        T_max = {r: [] for r in rooms}
        for i in range(horizon_steps):
            step_time = now + timedelta(minutes=i * STEP_MINUTES)
            step_local = step_time.astimezone(_BERLIN)
            hour = step_local.hour
            past_handoff = mode_end is not None and step_time >= mode_end
            office_hours = (7 <= hour < 18) and not past_handoff
            base_min, base_max = (22.0, 24.0) if office_hours else (20.0, 28.0)
            for r in rooms:
                T_min[r].append(base_min)
                T_max[r].append(base_max)
        return T_min, T_max

    @staticmethod
    def _tariff_schedule(horizon_steps: int, now: datetime) -> list:
        tariffs = []
        for i in range(horizon_steps):
            step_local = (now + timedelta(minutes=i * STEP_MINUTES)).astimezone(_BERLIN)
            tariffs.append(_REAL_TARIFF_BY_HOUR[step_local.hour])
        return tariffs

    @staticmethod
    def _sell_tariff_schedule(horizon_steps: int, now: datetime) -> list:
        tariffs = []
        for i in range(horizon_steps):
            step_local = (now + timedelta(minutes=i * STEP_MINUTES)).astimezone(_BERLIN)
            tariffs.append(_SELL_MIDDAY_DIP_EUR_KWH if step_local.hour in _SELL_MIDDAY_HOURS else _SELL_EUR_KWH)
        return tariffs
