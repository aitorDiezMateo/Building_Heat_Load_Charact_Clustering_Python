import numpy as np
import pandas as pd
import warnings

def Changepoint_Three_Parameters(slope_Temp, slope_Irrad, intercept, minimum, DF_input):

    Out = 0  # Control variable
    
    # Ensure pandas DataFrame (convert from Dask if needed)
    if not isinstance(DF_input, pd.DataFrame):
        try:
            DF_input = DF_input.compute()
        except Exception:
            warnings.warn("Input is not a pandas or Dask DataFrame; function did nothing")
            return DF_input

    if "Temperature" not in DF_input.columns:
        warnings.warn("DF missing 'Temperature' column")
        Out = 1
    if "Solar_Irradiation" not in DF_input.columns:
        warnings.warn("DF missing 'Solar_Irradiation' column")
        Out = 1
    if not isinstance(slope_Temp, (int, float, np.number)):
        warnings.warn("slope_Temp is not numeric")
        Out = 1
    if not isinstance(slope_Irrad, (int, float, np.number)):
        warnings.warn("slope_Irrad is not numeric")
        Out = 1
    if not isinstance(intercept, (int, float, np.number)):
        warnings.warn("intercept is not numeric")
        Out = 1

    if Out == 0 and "Power" not in DF_input.columns:
        warnings.warn("DF missing 'Power' column, residuals will not be computed")
        Out = 0.1
    
    DF_output = DF_input.copy()
    
    # --- Core computation ---
    if Out != 1:
        # Enforce non-negative minimum to avoid invalid thresholds
        if minimum < 0:
            warnings.warn("minimum < 0 detected; clamping to 0")
            minimum = 0.0

        temp_values = DF_input["Temperature"].to_numpy(dtype=np.float64, copy=False)
        irrad_values = DF_input["Solar_Irradiation"].to_numpy(dtype=np.float64, copy=False)

        output_values = intercept + slope_Temp * temp_values + slope_Irrad * irrad_values

        # Apply minimum threshold and non-negativity
        output_values = np.maximum(minimum, output_values)
        output_values = np.where(output_values >= 0, output_values, 0.0)

        DF_output = DF_output.assign(Power_fitted=output_values)

    # --- Residuals ---
    if Out == 0:
        DF_output = DF_output.assign(
            Power_residuals=DF_output["Power_fitted"] - DF_output["Power"]
        )
    elif Out == 0.1:
        DF_output = DF_output.assign(Power_residuals=np.nan)

    return DF_output

