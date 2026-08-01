import warnings
from datetime import datetime, timedelta, timezone

import dateutil
import pandas as pd
from wetterdienst.provider.dwd.mosmix.api import DwdMosmixRequest, DwdMosmixStationGroup
from wetterdienst.settings import Settings

_PARAM_RADIATION = "radiation_global"          # kJ/m² cumulative over last 1h
_PARAM_TEMPERATURE = "temperature_air_mean_2m"  # °C (wetterdienst converts from K)

# Any DWD MOSMIX ("small") forecast station works here. Pick whichever is
# nearest to the location you want to demo. Default below is an arbitrary
# public station (Berlin-Tempelhof), not tied to any real building.
DEFAULT_MOSMIX_STATION = "10382"

# wetterdienst's "energy_per_area" unit type defaults to J/cm² (the first entry in
# its internal unit list), NOT the kJ/m² that DWD documents for Rad1
_SETTINGS = Settings(ts_unit_targets={"energy_per_area": "kilojoule_per_square_meter"})


def get_forecast_weather(
        start_timestamp: datetime,
        end_timestamp: datetime,
        global_radiation_return_name="phi_s",
        global_radiation_unit="W/m^2",
        air_temperature_return_name="T_a",
        air_temperature_unit="°C",
        mosmix_station_index=DEFAULT_MOSMIX_STATION):

    # naive-UTC, matching start_timestamp/end_timestamp's convention (see the
    # .replace(tzinfo=...) below -- callers pass naive UTC wall-clock values)
    utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
    if end_timestamp > utc_now + timedelta(days=9, hours=23):
        warnings.warn("Warning: End too far in future")
    if start_timestamp < utc_now - timedelta(hours=1):
        warnings.warn("Warning: Start too far in past")
    if end_timestamp < start_timestamp:
        warnings.warn("Warning: End before start")

    forecast = DwdMosmixRequest(
        parameters=[
            ("hourly", "small", _PARAM_RADIATION),
            ("hourly", "small", _PARAM_TEMPERATURE),
        ],
        station_group=DwdMosmixStationGroup.SINGLE_STATIONS,
        start_date=start_timestamp,
        end_date=end_timestamp,
        settings=_SETTINGS,
    ).filter_by_station_id(mosmix_station_index)

    # wetterdienst >= 0.100 returns a Polars DataFrame in long (tidy) format.
    # Pivot to wide pandas DataFrame indexed by date.
    pl_df = forecast.values.all().df
    pdf = pl_df.to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"], utc=True)

    rad = pdf[pdf["parameter"] == _PARAM_RADIATION].set_index("date")["value"]
    tmp = pdf[pdf["parameter"] == _PARAM_TEMPERATURE].set_index("date")["value"]
    weather_frame = pd.concat([rad, tmp], axis=1)
    weather_frame.columns = [_PARAM_RADIATION, _PARAM_TEMPERATURE]

    # Unit conversions
    if global_radiation_unit == "W/m^2":
        # kJ/m² per 1-h interval → W/m² average over that hour
        weather_frame[_PARAM_RADIATION] = weather_frame[_PARAM_RADIATION] * 1000 / 3600
        # DWD Rad1h is "accumulated over the PAST 1h" but labelled at the END of the
        # interval, so shift -1 to align with the START of the next hour (forecast use).
        weather_frame[_PARAM_RADIATION] = weather_frame[_PARAM_RADIATION].shift(-1)

    # Clip to requested window (date index is UTC-aware)
    utc_start = start_timestamp.replace(tzinfo=dateutil.tz.UTC)
    utc_end = end_timestamp.replace(tzinfo=dateutil.tz.UTC)
    weather_frame = weather_frame.loc[utc_start:utc_end]

    weather_frame = weather_frame.tz_convert(None)
    weather_frame.index.names = ["start_time_utc"]
    weather_frame.columns = [global_radiation_return_name, air_temperature_return_name]

    return weather_frame


if __name__ == "__main__":
    now = datetime.now(timezone.utc).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    demo = get_forecast_weather(
        start_timestamp=now,
        end_timestamp=now + timedelta(hours=10),
    )
    print(demo)
