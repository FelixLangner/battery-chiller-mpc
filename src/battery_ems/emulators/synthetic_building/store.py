"""
TimeSeriesStore: stands in for "what InfluxDB would have recorded" in the real deployment.
A single time-indexed pandas.DataFrame stored to one parquet file.
"""
from pathlib import Path

import pandas as pd


class TimeSeriesStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.exists():
            self._df = pd.read_parquet(self.path)
        else:
            self._df = pd.DataFrame()
            self._df.index = pd.DatetimeIndex([], tz="UTC", name="time")

    def append(self, timestamp: pd.Timestamp, readings: dict) -> None:
        """Append one row of {signal_name: value} at `timestamp` (tz-aware UTC)."""
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        row = pd.DataFrame([readings], index=pd.DatetimeIndex([timestamp], name="time"))
        if self._df.empty:
            self._df = row
        else:
            self._df = pd.concat([self._df, row])
            self._df = self._df[~self._df.index.duplicated(keep="last")].sort_index()

    def query(self, columns: list[str], start: pd.Timestamp, end: pd.Timestamp,
              resolution: str = "5min") -> pd.DataFrame:
        """Range query, resampled -- the synthetic analogue of Influx's aggregateWindow."""
        if self._df.empty:
            return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], tz="UTC", name="time"))
        present = [c for c in columns if c in self._df.columns]
        sub = self._df.loc[(self._df.index >= start) & (self._df.index <= end), present]
        if sub.empty:
            return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], tz="UTC", name="time"))
        out = sub.resample(resolution).mean()
        for c in columns:
            if c not in out.columns:
                out[c] = float("nan")
        out.index.name = "time"
        return out[columns]

    def latest(self, columns: list[str]) -> dict:
        """Most recent value of each column, or NaN if the store is empty."""
        if self._df.empty:
            return {c: float("nan") for c in columns}
        last = self._df.iloc[-1]
        return {c: (float(last[c]) if c in last.index else float("nan")) for c in columns}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_parquet(self.path)

    def __len__(self) -> int:
        return len(self._df)
