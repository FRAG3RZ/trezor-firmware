from pathlib import Path
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from utils.console_formatter import ConsoleFormatter
from InquirerPy import inquirer
from InquirerPy.base import Choice
from dataset.battery_dataset import BatteryDataset
from dataset.battery_profile import time_to_minutes
from fuel_gauge.battery_model import BatteryModel

from fuel_gauge.battery_model import (
    load_battery_model_from_hash,
)
from fuel_gauge.simulator import run_battery_simulation
from fuel_gauge.coulomb_counter_estimator import CoulombCounterEstimator
from fuel_gauge.dummy_estimator import DummyEstimator
from fuel_gauge.ekf_estimator import EkfEstimator

DEBUG = False
BATTERY_MODEL_JSON_PATH = Path("exported_data/battery_models/")
DATASET_DIRECTORY = Path("dataset/datasets")
OUTPUT_DIRECTORY = Path("exported_data/simulation_results/")
SIMULATION_MODES = ["charging", "discharging", "random_wonder"]


# Global console formatter instance
console = ConsoleFormatter()

def prompt_for_dataset() -> BatteryDataset:
    """ Promt user to select a dataset from a DATASET_DIRECTORY and load it
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
    """ Prompt user to select a battery model from BATTERY_MODEL_JSON_PATH
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
    """ Generate a figure with the simulation results.
    Args:
        waveform: The waveform data used for the simulation.
        *sim_results: Variable number of simulation results to plot.
    Returns:
        matplotlib.figure.Figure: The generated figure with simulation results.
    """

    wd = waveform['data']

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
            time_to_minutes(sr.time[sr.start_idx:sr.end_idx]),
            sr.soc[sr.start_idx:sr.end_idx],
            label=sr.model_name,
        )
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

def run_simulation(dataset: BatteryDataset, battery_model: BatteryModel):

    console.section("Running battery simulation")
    # Create output directory based on battery model hash
    model_hash = battery_model.get_hash()
    output_dir = OUTPUT_DIRECTORY / model_hash
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

    # EKF parameters tuned for ~1s sampling rate
    ekf_est_1s = EkfEstimator(
        battery_model=battery_model,
        R=2000,
        Q=0.001,
        Q_agressive=0.001,
        R_agressive=1000,
        P_init=0.1,
        label="1s sampling"
    )
    console.info(f"EKF estimator 1s [R={ekf_est_1s.R}, Q={ekf_est_1s.Q}, "
                 f"Q_agressive={ekf_est_1s.Q_agressive}, R_agressive={ekf_est_1s.R_agressive}, "
                 f"P_init={ekf_est_1s.P_init}]")

    # EKF parameters scaled for 300ms sampling rate (FW target)
    ekf_est_300ms = EkfEstimator(
        battery_model=battery_model,
        R=2000,              # Measurement noise unchanged
        Q=0.0003,            # Process noise scaled: 0.001 * (300/1000)
        Q_agressive=0.0003,  # Aggressive process noise scaled
        R_agressive=1000,    # Measurement noise unchanged
        P_init=0.1,
        label="300ms sampling (FW)"
    )

    console.info(f"EKF estimator 300ms [R={ekf_est_300ms.R}, Q={ekf_est_300ms.Q}, "
                 f"Q_agressive={ekf_est_300ms.Q_agressive}, R_agressive={ekf_est_300ms.R_agressive},"
                 f"P_init={ekf_est_300ms.P_init}]")

    console.subsection("Running simulations")

    simulation_data = list(filtered_dataset.get_data_list())
    for idx, waveform in enumerate(simulation_data):

        data = waveform["data"]
        sim_name = f"{waveform['battery_id']}.{waveform['timestamp_id']}." \
                   f"{waveform['battery_mode']}.{waveform['mode_phase']}." \
                   f"{waveform['temperature']}"

        cc_result = run_battery_simulation(data, coulomb_counter_est)
        dm_result = run_battery_simulation(data, dummy_est)
        ekf_result_1s = run_battery_simulation(data, ekf_est_1s)
        ekf_result_300ms = run_battery_simulation(data, ekf_est_300ms)

        fig = generate_sim_res_fig(waveform, sim_name, cc_result, dm_result, ekf_result_1s, ekf_result_300ms)

        # Save as pickle format for reopening in Python
        with open(f"{pickle_dir / sim_name}.pkl", 'wb') as f:
            pickle.dump(fig, f)

        # Also save as PNG for viewing
        fig.savefig(f"{output_dir / sim_name}.png", dpi=300, bbox_inches='tight')

        # Close figure to free memory
        plt.close(fig)

        # Progress indicator
        console.progress(
            f" Simulation: {waveform['battery_id']}.{waveform['timestamp_id']}.{waveform['temperature']}°C",
            step=idx+1,
            total=len(simulation_data)
        )

def main():

    # Load simulation dataset
    console.header("Battery Simulation Tool")

    dataset = prompt_for_dataset()

    battery_model = prompt_for_battery_model()

    run_simulation(dataset, battery_model)



if __name__ == "__main__":
    main()