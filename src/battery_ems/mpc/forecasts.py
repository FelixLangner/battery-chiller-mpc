from datetime import datetime, timedelta, timezone

import pandas as pd
import pytz

from battery_ems.mpc.dwd import get_forecast_weather

""" Weather Forecast"""
# round to full hours
def round_dt(dt):
    return dt.replace(second=0, microsecond=0, minute=0)

def Forecasting(time_now, timestep, H):
    start = round_dt(time_now)
    end = round_dt(time_now) + timedelta(days=0, hours=H)

    # transform time into utc timezone
    local_timezone = pytz.timezone('Europe/Berlin')

    # Localize the datetime object
    localized_start = local_timezone.localize(start)
    localized_end = local_timezone.localize(end)

    # Convert to UTC
    utc_start = localized_start.astimezone(pytz.utc)
    utc_end = localized_end.astimezone(pytz.utc)

    # in the following functions, the timezone information is removed using
    # .replace(tzinfo=None) because the forecast functions require timezone
    # unaware datetime objects

    df_weather_forecast = get_forecast_weather(
        start_timestamp=utc_start.replace(tzinfo=None),
        end_timestamp=utc_end.replace(tzinfo=None),
        global_radiation_return_name="qs",
        global_radiation_unit="W/m^2",
        air_temperature_return_name="Ta",
        air_temperature_unit="°C")

    df_weather_forecast = df_weather_forecast.resample(str(timestep)+'min').interpolate(method='linear')
    Ta = df_weather_forecast["Ta"]
    qs = df_weather_forecast["qs"]
    # remove data that is older than 15 minutes (i.e., one time step)
    # naive-UTC, matching Ta.index's convention (dwd.py's tz_convert(None))
    utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
    time_threshold = utc_now - timedelta(minutes=15)

    # Check if the first value in the Series is in the future.
    # if so: copy this value for the previous time steps
    if Ta.index[0] > utc_now:
        new_index = pd.date_range(start=Ta.index[0]-timedelta(minutes=15), end=time_threshold, freq='-15min')
        Ta_new = pd.Series(data=[Ta.iloc[0]] * len(new_index), index=new_index)
        qs_new = pd.Series(data=[qs.iloc[0]] * len(new_index), index=new_index)
        Ta = pd.concat([Ta_new, Ta])
        qs = pd.concat([qs_new, qs])

    Ta = Ta[Ta.index >= time_threshold]
    qs = qs[qs.index >= time_threshold]

    return [qs, Ta]

if __name__ == "__main__":
    [qs, Ta] = Forecasting(datetime.now(), 5, 24)  # noqa: DTZ005 -- naive LOCAL time is the contract (see localize() above)
    print("Forecasting completed successfully.")
