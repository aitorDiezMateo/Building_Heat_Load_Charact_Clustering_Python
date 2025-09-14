import pandas as pd
import numpy as np

def date_marks(df_input, string_date, vec_vars):
    """
    Add date-related columns to the dataframe (equivalent to RG_Date_Marks in R)
    """
    df = df_input.copy()
    
    # Rename the date column to DATE for consistency
    if string_date != "DATE":
        df = df.rename(columns={string_date: "DATE"})
    
    # Ensure DATE is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['DATE']):
        df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Add date-related columns if they don't exist
    if 'DATE_year' not in df.columns:
        df['DATE_year'] = df['DATE'].dt.year
    if 'DATE_month_year' not in df.columns:
        df['DATE_month_year'] = df['DATE'].dt.month
    if 'DATE_week_year' not in df.columns:
        df['DATE_week_year'] = df['DATE'].dt.isocalendar().week
    if 'DATE_day_year' not in df.columns:
        df['DATE_day_year'] = df['DATE'].dt.dayofyear
    if 'DATE_day_month' not in df.columns:
        df['DATE_day_month'] = df['DATE'].dt.day
    if 'DATE_day_week' not in df.columns:
        df['DATE_day_week'] = df['DATE'].dt.dayofweek
    if 'DATE_hour_year' not in df.columns:
        df['DATE_hour_year'] = (df['DATE_day_year'] - 1) * 24 + df['DATE'].dt.hour
    if 'DATE_hour_week' not in df.columns:
        df['DATE_hour_week'] = df['DATE_day_week'] * 24 + df['DATE'].dt.hour
    if 'DATE_hour_day' not in df.columns:
        df['DATE_hour_day'] = df['DATE'].dt.hour
    if 'DATE_weekday' not in df.columns:
        df['DATE_weekday'] = df['DATE'].dt.weekday < 5
    
    return df


def repair_1h_double_side(df_input, string_date, vec_vars, vec_repair):
    """
    This function repairs 1h voids and outliers in the data frame
    given that:
    - only 1h void
    - data from previous and following week is available for the preceding and following hours
    
    Parameters:
    df_input: DataFrame - input data
    string_date: str - name of the column with datetime timestamp
    vec_vars: list - vector of columns with data
    vec_repair: list - vector of columns to repair
    
    Returns:
    DataFrame - repaired data
    """
    
    # Add date marks
    dat = date_marks(df_input, string_date, vec_vars)
    
    # Identification of missing data (& both missing or outlier data)
    dat['IS_Missing'] = (dat['Temperature'].isna() | 
                        dat['Solar_Irradiation'].isna() | 
                        dat['Power'].isna())
    
    # Handle IS_Outlier column
    if 'IS_Outlier' not in dat.columns:
        dat['IS_Outlier'] = False
    dat['IS_Outlier'] = dat['IS_Outlier'].fillna(True)
    
    dat['IS_Missing_Outlier'] = dat['IS_Missing'] | dat['IS_Outlier']
    
    if 'IS_Repaired' in dat.columns:
        dat['IS_Repair_Pending'] = dat['IS_Missing_Outlier'] & ~dat['IS_Repaired']
    else:
        dat['IS_Repair_Pending'] = dat['IS_Missing_Outlier']
    
    # Repair of 1h voids with data available for interpolation
    # Identification of missing spots
    # Use scipy.ndimage.uniform_filter1d to simulate R's stats::filter
    repair_pending_int = dat['IS_Repair_Pending'].astype(int)
    
    # Equivalent to stats::filter with c(0,1), sides=1 (forward filter)
    current_post1 = np.roll(repair_pending_int, -1)
    current_post1[-1] = 0  # Handle boundary
    
    # Equivalent to stats::filter with c(1,0), sides=2 (backward filter)
    current_pre1 = np.roll(repair_pending_int, 1)
    current_pre1[0] = 0  # Handle boundary
    
    dat['IS_Repair_Pending_single'] = (dat['IS_Repair_Pending'] & 
                                      ~current_post1.astype(bool) & 
                                      ~current_pre1.astype(bool))
    
    # Revision of data for previous & following week
    # Date marks
    dat['Date_currweek_prev_hour'] = dat['DATE'] - pd.Timedelta(hours=1)
    dat['Date_currweek_post_hour'] = dat['DATE'] + pd.Timedelta(hours=1)
    
    dat['Date_prevweek'] = dat['DATE'] - pd.Timedelta(weeks=1)
    dat['Date_prevweek_prev_hour'] = dat['Date_prevweek'] - pd.Timedelta(hours=1)
    dat['Date_prevweek_post_hour'] = dat['Date_prevweek'] + pd.Timedelta(hours=1)
    
    dat['Date_postweek'] = dat['DATE'] + pd.Timedelta(weeks=1)
    dat['Date_postweek_prev_hour'] = dat['Date_postweek'] - pd.Timedelta(hours=1)
    dat['Date_postweek_post_hour'] = dat['Date_postweek'] + pd.Timedelta(hours=1)
    
    # Are date marks available?
    good_dates = dat.loc[~dat['IS_Repair_Pending'], 'DATE']
    
    dat['IS_1h_interpol_possible'] = (
        dat['Date_currweek_prev_hour'].isin(good_dates) &
        dat['Date_currweek_post_hour'].isin(good_dates) &
        dat['Date_prevweek'].isin(good_dates) &
        dat['Date_prevweek_prev_hour'].isin(good_dates) &
        dat['Date_prevweek_post_hour'].isin(good_dates) &
        dat['Date_postweek'].isin(good_dates) &
        dat['Date_postweek_prev_hour'].isin(good_dates) &
        dat['Date_postweek_post_hour'].isin(good_dates)
    )
    
    # Get smaller subset
    dat_sbs = dat[dat['IS_1h_interpol_possible'] & dat['IS_Repair_Pending_single']].copy()
    
    if len(dat_sbs) > 0:
        # Linear interpolation
        def get_data_for_dates(dates, columns):
            return dat.loc[dat['DATE'].isin(dates), columns]
        
        # Get data for all required time points
        data_currweek = get_data_for_dates(dat_sbs['DATE'], vec_repair)
        data_currweek_prev_hour = get_data_for_dates(dat_sbs['Date_currweek_prev_hour'], vec_repair)
        data_currweek_post_hour = get_data_for_dates(dat_sbs['Date_currweek_post_hour'], vec_repair)
        
        data_prevweek = get_data_for_dates(dat_sbs['Date_prevweek'], vec_repair)
        data_prevweek_prev_hour = get_data_for_dates(dat_sbs['Date_prevweek_prev_hour'], vec_repair)
        data_prevweek_post_hour = get_data_for_dates(dat_sbs['Date_prevweek_post_hour'], vec_repair)
        
        data_postweek = get_data_for_dates(dat_sbs['Date_postweek'], vec_repair)
        data_postweek_prev_hour = get_data_for_dates(dat_sbs['Date_postweek_prev_hour'], vec_repair)
        data_postweek_post_hour = get_data_for_dates(dat_sbs['Date_postweek_post_hour'], vec_repair)
        
        # Calculate interpolated values using the same formula as R
        interpolated_values = (
            ((data_prevweek[vec_repair].values + data_postweek[vec_repair].values) / 2) +
            ((data_currweek_prev_hour[vec_repair].values + data_currweek_post_hour[vec_repair].values) / 2) -
            ((data_prevweek_prev_hour[vec_repair].values + data_prevweek_post_hour[vec_repair].values +
              data_postweek_prev_hour[vec_repair].values + data_postweek_post_hour[vec_repair].values) / 4)
        )
        
        # Create corrected column names
        vec_repair_corrected = [col + '_corrected' for col in vec_repair]
        
        # Add corrected columns to the main dataframe
        for i, col in enumerate(vec_repair):
            corrected_col = col + '_corrected'
            if corrected_col not in dat.columns:
                dat[corrected_col] = dat[col].copy()
            
            # Update the corrected values for the repaired rows
            mask = dat['DATE'].isin(dat_sbs['DATE'])
            dat.loc[mask, corrected_col] = interpolated_values[:, i]
    
    else:
        # No data to repair, just create corrected columns as copies
        vec_repair_corrected = [col + '_corrected' for col in vec_repair]
        for col in vec_repair:
            corrected_col = col + '_corrected'
            if corrected_col not in dat.columns:
                dat[corrected_col] = dat[col].copy()
    
    # Update IS_Repaired column
    if 'IS_Repaired' in dat.columns:
        dat['IS_Repaired'] = dat['IS_Repaired'] | (dat['IS_Repair_Pending'] & dat['IS_1h_interpol_possible'])
    else:
        dat['IS_Repaired'] = dat['IS_Repair_Pending'] & dat['IS_1h_interpol_possible']
    
    # Select output columns
    varnames_out = [string_date if string_date != "DATE" else "DATE",
                    "DATE_year", "DATE_month_year", "DATE_week_year", "DATE_day_year",
                    "DATE_day_month", "DATE_day_week", "DATE_hour_year", "DATE_hour_week",
                    "DATE_hour_day", "DATE_weekday"] + vec_vars + [
                    "IS_Missing", "IS_Missing_Outlier", "IS_Repaired"] + vec_repair_corrected
    
    # Keep only columns that exist in the dataframe
    existing_cols = [col for col in varnames_out if col in dat.columns]
    dat = dat[existing_cols]
    
    return dat
