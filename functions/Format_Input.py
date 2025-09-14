import dask.dataframe as dd

def format_input(df: dd.DataFrame,
                 col_date: str = "Date",
                 col_holiday: str = "Holiday",
                 col_temperature: str = "Temperature",
                 col_solar_irradiation: str = "Solar.Irradiation",
                 col_power: str = "Power") -> dd.DataFrame:
    
    df = df.rename(columns={col_date: "DATE",
                            col_holiday: "Holiday",
                            col_temperature: "Temperature",
                            col_solar_irradiation: "Solar_Irradiation",
                            col_power: "Power"})
    
    # Keep only necessary columns
    df = df[["DATE", "Holiday", "Temperature", "Solar_Irradiation", "Power"]]
    
    df = df.assign(DATE=dd.to_datetime(df.DATE))
    
    df = df.assign(Holiday=df.Holiday.astype(bool))
    
    df = df.assign(Temperature=df.Temperature.astype(float))
    
    df = df.assign(Solar_Irradiation=df["Solar_Irradiation"].astype(float))
    
    df = df.assign(Power=df.Power.astype(float))
    
    # DATE_YYYY_MM_DD
    df = df.assign(DATE_YYYY_MM_DD = df.DATE.dt.date)
    # DATE_year
    df = df.assign(DATE_year = df.DATE.dt.year)
    # DATE_month_year
    df = df.assign(DATE_month_year = df.DATE.dt.month)
    # DATE_week_year
    df = df.assign(DATE_week_year = df.DATE.dt.isocalendar().week)
    # DATE_day_year
    df = df.assign(DATE_day_year = df.DATE.dt.dayofyear)
    # DATE_day_month
    df = df.assign(DATE_day_month = df.DATE.dt.day)
    # DATE_day_week
    df = df.assign(DATE_day_week = df.DATE.dt.dayofweek)
    # DATE_hour_year
    df = df.assign(DATE_hour_year = (df["DATE_day_year"]-1)*24 + df.DATE.dt.hour)
    # DATE_hour_week
    df = df.assign(DATE_hour_week = (df["DATE_day_week"])*24 + df.DATE.dt.hour)
    # DATE_hour_day
    df = df.assign(DATE_hour_day = df.DATE.dt.hour)
    # DATE_weekday (True for Monday-Friday, False for Saturday-Sunday)
    df = df.assign(DATE_weekday = df.DATE.dt.weekday < 5)
    
    return df