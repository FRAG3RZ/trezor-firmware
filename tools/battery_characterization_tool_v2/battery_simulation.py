#!/usr/bin/env python3

import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataset.battery_dataset import BatteryDataset
from dataset.battery_profile import time_to_minutes
from InquirerPy import inquirer
from InquirerPy.base import Choice
from matplotlib.figure import Figure
from models.battery_model import BatteryModel, load_battery_model_from_hash
from models.estimators import CoulombCounterEstimator, DummyEstimator, EkfEstimator
from models.simulator import run_battery_simulation
from utils.console_formatter import ConsoleFormatter
from utils.error_functions import summarize_simulation_errors

DEBUG = False
BATTERY_MODEL_JSON_PATH = Path("exported_data/battery_models/")
DATASET_DIRECTORY = Path("dataset/datasets")
OUTPUT_DIRECTORY = Path("exported_data/simulation_results/")
SIMULATION_MODES = ["charging", "discharging", "random_wonder"]

# Global console formatter instance
console = ConsoleFormatter()

def prompt_for_dataset() -> BatteryDataset:
    """Promt user to select a dataset from a DATASET_DIRECTORY and load it
        into a BatteryDataset object.
    Returns:
        BatteryDataset: The selected dataset object.
    """

    console.section("Dataset Selection")

    dataset_folders = [f.name for f in DATASET_DIRECTORY.iterdir() if f.is_dir()]

    if not dataset_folders:
        raise FileNotFoundError("ERROR: No dataset folders found in 'dataset/datasets'")

    choices = [Choice(name=name, value=name) for name in dataset_folders]

    selected_folder = inquirer.select(
        message="Choose a dataset folder to load:",
        choices=choices,
    ).execute()

    dataset_path = DATASET_DIRECTORY / selected_folder
    battery_dataset = BatteryDataset(dataset_path, load_data=True)

    console.success(f"Selected dataset: {dataset_path}")

    return battery_dataset


def prompt_for_battery_model() -> BatteryModel:
    """Prompt user to select a battery model from BATTERY_MODEL_JSON_PATH
        and load it into a BatteryModel object.
    Returns:
        BatteryModel: The selected battery model object.
    """

    console.section("Battery Model Selection")

    json_files = sorted(BATTERY_MODEL_JSON_PATH.glob("*.json"))
    if not json_files:
        console.error(f"No .json files found in {BATTERY_MODEL_JSON_PATH}")
        raise FileNotFoundError("No battery model files found.")

    choices = [Choice(name=file.name, value=file) for file in json_files]

    selected_file = inquirer.select(
        message="Choose a battery model to load:",
        choices=choices,
    ).execute()

    file_path = selected_file

    console.info(f"Selected file path: {file_path}")

    battery_model = load_battery_model_from_hash(file_path)

    console.success(f"Loaded battery model with hash: {battery_model.model_hash}")

    return battery_model

def generate_sim_res_fig(waveform, sim_name, *sim_results) -> Figure:
    """Generate a figure with the simulation results.
    Args:
        waveform: The waveform data used for the simulation.
        *sim_results: Variable number of simulation results to plot.
    Returns:
        matplotlib.figure.Figure: The generated figure with simulation results.
    """

    wd = waveform["data"]

    fig, ax = plt.subplots(4, 1)
    fig.set_size_inches(10, 10)
    fig.suptitle(f"{sim_name}°C")

    ax[0].plot(time_to_minutes(wd.time), wd.vbat, label="vbat")
    ax[0].set_title("Battery voltage")
    ax[0].set_ylabel("Voltage [V]")
    ax[0].set_xlabel("Time [min]")
    ax[0].set_xlim((0, time_to_minutes(wd.time[-1], wd.time[0])))

    ax[1].plot(time_to_minutes(wd.time), wd.ibat)
    ax[1].set_title("Battery current")
    ax[1].set_ylabel("Current [mA]")
    ax[1].set_xlabel("Time [min]")
    ax[1].set_xlim((0, time_to_minutes(wd.time[-1], wd.time[0])))

    for sr in sim_results:
        ax[2].plot(
            time_to_minutes(sr.time[sr.start_idx: sr.end_idx]),
            sr.soc[sr.start_idx: sr.end_idx],
            label=sr.model_name,
        )
    
    #Plot linear deviation line if available
    for sr in sim_results:
        if sr.deviation_error is not None:
            x_start = sr.deviation_error.get("t_start")
            y_start = sr.deviation_error.get("soc_start")
            x_end = sr.deviation_error.get("t_stop")
            y_end = sr.deviation_error.get("soc_end")

            #print(f"Linear deviation line coordinates: ({time_to_minutes(x_start, wd.time[0])}, {y_start}) -> ({time_to_minutes(x_end, wd.time[0])}, {y_end})")

            # Only plot if all coordinates are valid numbers
            if None not in (x_start, y_start, x_end, y_end):
                ax[2].plot(
                    [time_to_minutes(x_start, wd.time[0]), time_to_minutes(x_end, wd.time[0])],
                    [y_start, y_end],
                    color="purple",
                    linestyle="-",
                    linewidth=2,
                    label=f"Linear deviation line - cummulative error: {sr.deviation_error.get('cumulative_error', 'N/A')}"
                )
            else:
                print("Skipping linear deviation line due to missing values")

            break  # Only plot once

    ax[2].set_title("Estimated SoC")
    ax[2].set_xlabel("Time [min]")
    ax[2].set_ylabel("SoC [%]")
    ax[2].set_xlim([0, time_to_minutes(wd.time[-1], wd.time[0])])
    ax[2].legend()

    for h in [0, 0.25, 0.5, 0.75, 1.0]:
        ax[2].hlines(h, *ax[2].get_xlim(), color="black", alpha=0.3)

    ax[3].plot(time_to_minutes(wd.time), wd.ntc_temp)
    ax[3].set_title("ntc temperature")
    ax[3].set_xlabel("Time [min]")
    ax[3].set_ylabel("Temperature [°C]")
    ax[3].set_xlim((0, time_to_minutes(wd.time[-1], wd.time[0])))
 
    return fig

def run_simulations_for_R_Q(dataset, battery_model, R_min, R_max, R_count, Q_min, Q_max, Q_count):
    R_values = np.linspace(R_min, R_max, R_count)
    Q_values = np.linspace(Q_min, Q_max, Q_count)

    total_runs = len(R_values) * len(Q_values)
    run_count = 0

    for R in R_values:
        for Q in Q_values:
            run_count += 1
            console.progress(
                f"Running R-Q grid search → R={R:.5f}, Q={Q:.5f}",
                step=run_count,
                total=total_runs,
            )
            run_simulation(dataset, battery_model, R, Q, R, Q, P_init=0.1)

def run_simulation(dataset: BatteryDataset, battery_model: BatteryModel, R_value, Q_value, R_aggressive=1000, Q_aggressive=0.001, P_init=0.1):

    error_results = []

    console.section("Running battery simulation")
    # Create output directory based on battery model hash
    model_hash = battery_model.get_hash()
    output_dir = OUTPUT_DIRECTORY / f"{model_hash}_R-{R_value}_Q-{Q_value}"
    output_dir.mkdir(parents=True, exist_ok=True)
    pickle_dir = output_dir / "pickles"
    pickle_dir.mkdir(parents=True, exist_ok=True)

    console.subsection(f"Filter eligible datasets")
    console.info(f"Filter datasets with mode_phases {SIMULATION_MODES}")
    filtered_dataset = dataset.filter(mode_phases=SIMULATION_MODES)
    console.success(f"Filtered dataset statistics: {filtered_dataset.get_statistics()}")
    console.subsection("Preparing estimation models")

    coulomb_counter_est = CoulombCounterEstimator(battery_model=battery_model)
    console.info("Coulomb counter estimator")

    dummy_est = DummyEstimator(battery_model=battery_model)
    console.info("Dummy estimator")

    ekf_est = EkfEstimator(
        battery_model=battery_model,
        R=R_value,
        Q=Q_value,
        Q_agressive=Q_aggressive,
        R_agressive=R_aggressive,
        P_init=P_init,
    )
    console.info(
        f"EKF estimator [R={ekf_est.R}, Q={ekf_est.Q}, "
        f"Q_agressive={ekf_est.Q_agressive}, R_agressive={ekf_est.R_agressive}, "
        f"P_init={ekf_est.P_init}]"
    )

    ekf_est2 = EkfEstimator(
        battery_model=battery_model,
        R=R_value,
        Q=Q_value,
        Q_agressive=Q_aggressive,
        R_agressive=R_aggressive,
        P_init=P_init,
    )

    console.info(
        f"EKF estimator 2 [R={ekf_est2.R}, Q={ekf_est2.Q}, "
        f"Q_agressive={ekf_est2.Q_agressive}, R_agressive={ekf_est2.R_agressive},"
        f"P_init={ekf_est2.P_init}]"
    )

    console.subsection("Running simulations")

    simulation_data = list(filtered_dataset.get_data_list())
    for idx, waveform in enumerate(simulation_data):

        data = waveform["data"]
        sim_name = (
            f"{waveform['battery_id']}.{waveform['timestamp_id']}."
            f"{waveform['battery_mode']}.{waveform['mode_phase']}."
            f"{waveform['temperature']}"
        )

        #Fallback for erorr appending in case of not running ekf2 simulation
        ekf_result = None
        ekf2_result = None

        ekf_result = run_battery_simulation(data, ekf_est, battery_model=battery_model, sim_name=sim_name)

        """
        ekf2_result = run_battery_simulation(data, ekf_est2, battery_model=battery_model, sim_name=sim_name)
        cc_result = run_battery_simulation(data, coulomb_counter_est, battery_model=battery_model, sim_name=sim_name)
        dm_result = run_battery_simulation(data, dummy_est, battery_model=battery_model, sim_name=sim_name)
        
        fig = generate_sim_res_fig(
            waveform, sim_name, cc_result, dm_result, ekf_result, ekf2_result
        )

        # Also save as PNG for viewing
        fig.savefig(f"{output_dir / sim_name}.png", dpi=300, bbox_inches="tight")

        # Save as pickle format for reopening in Python
        with open(f"{pickle_dir / sim_name}.pkl", "wb") as f:
            pickle.dump(fig, f)

        # Close figure to free memory
        plt.close(fig)
        """
        if battery_model is not None:
            slope_err_ekf1 = ekf_result.error if ekf_result.error is not None else None 
            slope_err_ekf2 = ekf2_result.error if ekf2_result.error is not None else None
            end_soc_err_ekf1 = ekf_result.error_soc if ekf_result.error_soc is not None else None
            end_soc_err_ekf2 = ekf2_result.error_soc if ekf2_result.error_soc is not None else None
            deviation_error_ekf1 = ekf_result.deviation_error if ekf_result.deviation_error is not None else None
            deviation_error_ekf2 = ekf2_result.deviation_error if ekf2_result.deviation_error is not None else None

            error_results.append({
                "file": sim_name,
                "R": R_value,
                "R_aggressive": R_aggressive,
                "Q": Q_value,
                "Q_aggressive": Q_aggressive,

                # Slope errors (EKF1)
                "avg_ekf1_slope_error": slope_err_ekf1["avg_error"] if slope_err_ekf1 is not None else None,
                "rms_ekf1_slope_error": slope_err_ekf1["rms_error"] if slope_err_ekf1 is not None else None,
                "max_ekf1_slope_error": slope_err_ekf1["max_error"] if slope_err_ekf1 is not None else None,
                "ekf1_slope_error": slope_err_ekf1["cumulative_error"] if slope_err_ekf1 is not None else None,
                "valid_points_ekf1_slope": slope_err_ekf1["valid_points"] if slope_err_ekf1 is not None else None,

                # Slope errors (EKF2)
                "avg_ekf2_slope_error": slope_err_ekf2["avg_error"] if slope_err_ekf2 is not None else None,
                "rms_ekf2_slope_error": slope_err_ekf2["rms_error"] if slope_err_ekf2 is not None else None,
                "max_ekf2_slope_error": slope_err_ekf2["max_error"] if slope_err_ekf2 is not None else None,
                "ekf2_slope_error": slope_err_ekf2["cumulative_error"] if slope_err_ekf2 is not None else None,
                "valid_points_ekf2_slope": slope_err_ekf2["valid_points"] if slope_err_ekf2 is not None else None,

                # End-of-cycle errors (EKF1)
                "ekf1_end_soc_error": end_soc_err_ekf1 if end_soc_err_ekf1 is not None else None,

                # End-of-cycle errors (EKF2)
                "ekf2_end_soc_error": end_soc_err_ekf2 if end_soc_err_ekf2 is not None else None,

                # Deviation error (EKF1)
                "ekf1_deviation_error": deviation_error_ekf1["cumulative_error"] if deviation_error_ekf1 is not None else None,
                "mean_ekf1_deviation_error": deviation_error_ekf1["mean_error"] if deviation_error_ekf1 is not None else None,

                # Deviation error (EKF2)
                "ekf2_deviation_error": deviation_error_ekf2["cumulative_error"] if deviation_error_ekf2 is not None else None,
                "mean_ekf2_deviation_error": deviation_error_ekf2["mean_error"] if deviation_error_ekf2 is not None else None
            })
        
        # Progress indicator
        console.progress(
            f" Simulation: {waveform['battery_id']}.{waveform['timestamp_id']}.{waveform['temperature']}°C",
            step=idx + 1,
            total=len(simulation_data),
        )
        
    # Save all results to CSV
    results_df = pd.DataFrame(error_results)
    results_csv = OUTPUT_DIRECTORY / f"error_results_R-{R_value}_Q-{Q_value}.csv"
    results_df.to_csv(results_csv, index=False)

    console.success("Simulation completed.")

def main():

    # Load simulation dataset
    console.header("Battery Simulation Tool")

    dataset = prompt_for_dataset()

    battery_model = prompt_for_battery_model()

    #run_simulation(dataset, battery_model, 1800, 0.004)
    #run_simulation(dataset, battery_model, 1800, 0.005)
    #run_simulation(dataset, battery_model, 1700, 0.004)
    #run_simulation(dataset, battery_model, 1700, 0.005)

    #run_simulations_for_R_Q(dataset, battery_model, R_min=100, R_max=900, R_count=9, Q_min=0.0015, Q_max=0.0015, Q_count=1)
    #run_simulations_for_R_Q(dataset, battery_model, R_min=100, R_max=900, R_count=9, Q_min=0.00075, Q_max=0.00075, Q_count=1)
    #run_simulations_for_R_Q(dataset, battery_model, R_min=100, R_max=900, R_count=9, Q_min=0.0005, Q_max=0.0005, Q_count=1)
    #run_simulations_for_R_Q(dataset, battery_model, R_min=100, R_max=900, R_count=9, Q_min=0.001, Q_max=0.005, Q_count=5)

    console.footer

    summarize_simulation_errors(OUTPUT_DIRECTORY)

if __name__ == "__main__":
    main()