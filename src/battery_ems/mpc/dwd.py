import warnings
from datetime import datetime, timedelta

import dateutil
import pandas as pd
from wetterdienst.provider.dwd.mosmix.api import DwdMosmixRequest, DwdMosmixStationGroup
from wetterdienst.provider.dwd.observation import DwdObservationRequest
from wetterdienst.settings import Settings

_PARAM_RADIATION = "radiation_global"          # kJ/m² cumulative over last 1h
_PARAM_TEMPERATURE = "temperature_air_mean_2m"  # °C (wetterdienst converts from K)

# Any DWD MOSMIX ("small") forecast station works here -- pick whichever is
# nearest to the location you want to demo. Default below is an arbitrary
# public station, not tied to any real building.
DEFAULT_MOSMIX_STATION = "10731"
# Paired historical-observation station for get_historical_weather() (used
# by the optional Chronos PV/load forecasting path, not the cooling MPC).
DEFAULT_OBSERVATION_STATION = "04177"

# wetterdienst's "energy_per_area" unit type defaults to J/cm² (the first entry in
# its internal unit list), NOT the kJ/m² that DWD documents for Rad1h -- silently
# returning values 10x smaller than the conversion below assumes. Confirmed against
# the official DWD MOSMIX_L KML vs wetterdienst's default output for the same
# station/hour. Pin the target unit explicitly rather than relying on wetterdienst's
# default, which could change.
_SETTINGS = Settings(ts_unit_targets={"energy_per_area": "kilojoule_per_square_meter"})


def get_forecast_weather(
        start_timestamp: datetime,
        end_timestamp: datetime,
        global_radiation_return_name="phi_s",
        global_radiation_unit="W/m^2",
        air_temperature_return_name="T_a",
        air_temperature_unit="°C",
        mosmix_station_index=DEFAULT_MOSMIX_STATION):

    if end_timestamp > datetime.utcnow() + timedelta(days=9, hours=23):
        warnings.warn("Warning: End too far in future")
    if start_timestamp < datetime.utcnow() - timedelta(hours=1):
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


def get_historical_weather(
        start_timestamp: datetime,
        end_timestamp: datetime,
        resample_freq: str = "15min",
        station_id: str = DEFAULT_OBSERVATION_STATION,
) -> pd.DataFrame:
    """
    Past (measured, not forecast) global radiation (W/m²) and air temperature
    (°C) from DWD's observation network, resampled to a regular
    `resample_freq` grid. Returns columns ["DWD_GlobIrradHoriz", "DWD_EnvTmp"].
    """
    settings = Settings(ts_unit_targets={"energy_per_area": "kilojoule_per_square_meter"})
    request = DwdObservationRequest(
        parameters=[
            ("minute_10", "solar", _PARAM_RADIATION),
            ("hourly", "temperature_air", _PARAM_TEMPERATURE),
        ],
        start_date=start_timestamp,
        end_date=end_timestamp,
        settings=settings,
    ).filter_by_station_id(station_id)

    pdf = request.values.all().df.to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"], utc=True)

    rad = pdf[pdf["parameter"] == _PARAM_RADIATION].set_index("date")["value"]
    tmp = pdf[pdf["parameter"] == _PARAM_TEMPERATURE].set_index("date")["value"]

    # kJ/m² per 10-min interval -> W/m² average power over that interval.
    rad_w = (rad * 1000.0 / 600.0).rename("DWD_GlobIrradHoriz")
    tmp_c = tmp.rename("DWD_EnvTmp")

    weather = pd.concat([rad_w, tmp_c], axis=1)
    weather = weather.resample(resample_freq).interpolate(method="linear")
    utc_start = start_timestamp if start_timestamp.tzinfo else start_timestamp.replace(tzinfo=dateutil.tz.UTC)
    utc_end = end_timestamp if end_timestamp.tzinfo else end_timestamp.replace(tzinfo=dateutil.tz.UTC)
    utc_start = pd.Timestamp(utc_start).tz_convert("UTC")
    utc_end = pd.Timestamp(utc_end).tz_convert("UTC")
    weather = weather.loc[utc_start:utc_end]
    weather["DWD_GlobIrradHoriz"] = weather["DWD_GlobIrradHoriz"].clip(lower=0.0)
    return weather


if __name__ == "__main__":
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    demo = get_forecast_weather(
        start_timestamp=now,
        end_timestamp=now + timedelta(hours=10),
    )
    print(demo)
