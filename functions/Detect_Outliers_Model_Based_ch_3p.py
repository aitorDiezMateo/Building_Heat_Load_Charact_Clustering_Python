import pandas as pd
from deap import base, creator, tools, algorithms
import sys
import os
import numpy as np
import random
import warnings
from joblib import Parallel, delayed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.Changepoint_Three_Parameters import Changepoint_Three_Parameters


def Changepoint_Three_Parameters_OBJECTIVE_FUN(slope_Temp, slope_Irrad, intercept, minimum, DF_input):
    # Ensure pandas DataFrame
    if not isinstance(DF_input, pd.DataFrame):
        DF_input = DF_input.compute()

    result_df = Changepoint_Three_Parameters(
        slope_Temp, slope_Irrad, intercept, minimum, DF_input
    )

    residuals = result_df["Power_residuals"].to_numpy(dtype=np.float64, copy=False)
    residuals = residuals[~np.isnan(residuals)]
    fitness = np.sum(np.abs(residuals))

    return fitness

def Detect_Outliers_Model_Based_CH_3P(DF_input, threshold_outlier=2, pop_size=100, max_iter=100, verbose=False,slurm_cluster=False):
    # Ensure pandas DataFrame for fast scalar ops
    if not isinstance(DF_input, pd.DataFrame):
        DF_input = DF_input.compute()

    # Step 1: Compute scalar stats from pandas
    irrad_max = DF_input["Solar_Irradiation"].max(skipna=True)
    if pd.isna(irrad_max) or irrad_max == 0:
        irrad_max = 1.0

    power_max = DF_input["Power"].max(skipna=True)
    temp_max = DF_input["Temperature"].max(skipna=True)
    temp_min = DF_input["Temperature"].min(skipna=True)

    temp_range = temp_max - temp_min if not pd.isna(temp_max) and not pd.isna(temp_min) else 0.0
    slope_Temp_MIN = -power_max / temp_range if temp_range != 0 else -power_max
    slope_Irrad_MIN = -power_max / irrad_max if irrad_max != 0 else -power_max
    intercept_MAX = power_max
    minimum_MAX = power_max
    
    if verbose:
        print("\n=== Evolutionary algorithm started ===\n")

    
    if verbose:
        print("Scalar stats:")
        print("slope_Temp_MIN:", slope_Temp_MIN)
        print("slope_Irrad_MIN:", slope_Irrad_MIN)
        print("intercept_MAX:", intercept_MAX)
        print("minimum_MAX:", minimum_MAX)
        print("\n") 
    
    if minimum_MAX < 0:
        warnings.warn("minimum_MAX < 0 detected; clamping to 0")
        minimum_MAX = 0.0
    
    # If all Power values are zero (or near zero)
    if (
        slope_Temp_MIN + slope_Irrad_MIN + intercept_MAX + minimum_MAX == 0
        or power_max == 0
    ):
        DF_Output = Changepoint_Three_Parameters(0, 0, 0, 0, DF_input)
        DF_Output["IS_Outlier"] = False
        return DF_Output, [0, 0, 0, 0]
    
    # Parallel map with joblib
    def joblib_map(func, iterable):
        if slurm_cluster:
            n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
        else:
            n_cores = -1
        return Parallel(n_jobs=n_cores)(delayed(func)(item) for item in iterable)

    # Step 2: Define fitness function
    def fitness_function(individual):
        slope_Temp, slope_Irrad, intercept, minimum = individual
        return (
            Changepoint_Three_Parameters_OBJECTIVE_FUN(
                slope_Temp, slope_Irrad, intercept, minimum, DF_input
            ),
        )
    
    # Explicit bounds for each gene: [slope_Temp, slope_Irrad, intercept, minimum]
    BOUND_LOW = [slope_Temp_MIN, slope_Irrad_MIN, 0.0, 0.0]
    BOUND_UP = [0.0, 0.0, intercept_MAX, minimum_MAX]

    # Step 3: Genetic Algorithm setup
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_slopeTemp", random.uniform, slope_Temp_MIN, 0)
    toolbox.register("attr_slopeIrrad", random.uniform, slope_Irrad_MIN, 0)
    toolbox.register("attr_intercept", random.uniform, 0, intercept_MAX)
    toolbox.register("attr_minimum", random.uniform, 0, minimum_MAX)

    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (toolbox.attr_slopeTemp, toolbox.attr_slopeIrrad, toolbox.attr_intercept, toolbox.attr_minimum),
        n=1,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fitness_function)
    
    # Use joblib's parallel map
    toolbox.register("map", joblib_map)
    
    # Bounded crossover and mutation to keep genes within [low, up]
    toolbox.register(
        "mate",
        tools.cxSimulatedBinaryBounded,
        low=BOUND_LOW,
        up=BOUND_UP,
        eta=15.0,
    )
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=BOUND_LOW,
        up=BOUND_UP,
        eta=20.0,
        indpb=0.2,
    )
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Step 4: Run GA
    population = toolbox.population(n=pop_size)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=max_iter, verbose=False)


    best_params = tools.selBest(population, 1)[0]
    slope_Temp_OPT, slope_Irrad_OPT, intercept_OPT, minimum_OPT = best_params
    
    if verbose:
        print("Best individual found:")
        print("slope_Temp_OPT:", slope_Temp_OPT)
        print("slope_Irrad_OPT:", slope_Irrad_OPT)
        print("intercept_OPT:", intercept_OPT)
        print("minimum_OPT:", minimum_OPT)
    
    if verbose:
        print("\n=== Evolutionary algorithm finished ===\n")
    
    # Step 5: Build output
    DF_Output = Changepoint_Three_Parameters(
        slope_Temp_OPT, slope_Irrad_OPT, intercept_OPT, minimum_OPT, DF_input
    )

    residuals = DF_Output["Power_residuals"]
    DF_Output["IS_Outlier"] = (
        (residuals - residuals.mean()).abs() > threshold_outlier * residuals.std()
    )

    return DF_Output, best_params