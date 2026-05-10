import pandas as pd
import pandas_market_calendars as pmc
from datetime import datetime, timedelta


def get_weekly_steps(start_date: str, end_date: str, output_path: str = None):
    nyse = pmc.get_calendar("XNYS")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    expanded_start_dt = start_dt - timedelta(days=7)
    expanded_start_date = expanded_start_dt.strftime("%Y-%m-%d")
    schedule = nyse.schedule(
        start_date=expanded_start_date,   
        end_date=end_date)

    sched = schedule.copy()
    sched["session_date"] = sched.index.date
    sched["close_utc"] = sched["market_close"].dt.tz_convert("UTC")
    sched = sched[["session_date", "close_utc"]].reset_index(drop=True)
    sched["session_date"] = pd.to_datetime(sched["session_date"]) 
    sched["week_start_monday"] = (
        sched["session_date"] - pd.to_timedelta(sched["session_date"].dt.weekday, unit="D")
    )

    weekly_last = (
        sched.sort_values("session_date")
            .groupby("week_start_monday", as_index=False)
            .tail(1)  # last trading day of that week
            .copy())

    # keep only the relevant columns
    weekly_last = weekly_last[["session_date", "close_utc"]]
    weekly_last = weekly_last.sort_values("session_date").reset_index(drop=True)

    weekly_last = weekly_last.rename(columns={
        "session_date": "week_decision_date",
        "close_utc": "curr_close_utc"})

    weekly_last["prev_close_utc"] = weekly_last["curr_close_utc"].shift(1)
    weekly_last = weekly_last.iloc[1:].reset_index(drop=True)
    weekly_last["week_decision_date"] = pd.to_datetime(weekly_last["week_decision_date"]).dt.strftime("%Y-%m-%d")
    weekly_last["curr_close_utc"] = weekly_last["curr_close_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    weekly_last["prev_close_utc"] = weekly_last["prev_close_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    if output_path is not None:
        weekly_last.to_parquet(output_path, index=False)
    return weekly_last

