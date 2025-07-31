from pathlib import Path
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from InquirerPy import inquirer
from InquirerPy.base import Choice
from utils.data_convertor import (
    load_measured_data_new as load_measured_data,
)
from fuel_gauge.battery_profiling import (
    rational_fit,
    rational_linear_rational,
)
from fuel_gauge.battery_model import (
    load_battery_model_from_hash,
)
from fuel_gauge.simulator import run_battery_simulation
from fuel_gauge.coulomb_counter_estimator import CoulombCounterEstimator
from fuel_gauge.dummy_estimator import DummyEstimator
from fuel_gauge.ekf_estimator import EkfEstimator
from fuel_gauge.fuel_gauge import fuel_gauge
from fuel_gauge.battery_profiling import time_to_minutes, low_pass_ma_filter

# ==========Configuration==========

DEBUG = False

json_path = Path("exported_libraries_updated/battery_models_jsons")

def prompt_for_dataset_folder():
    dataset_dir = Path("dataset")
    dataset_folders = [f.name for f in dataset_dir.iterdir() if f.is_dir()]

    if not dataset_folders:
        raise FileNotFoundError("❌ No dataset folders found in 'dataset/'")

    print("\n📁 Available dataset folders:")
    for i, name in enumerate(dataset_folders):
        print(f"  [{i}] {name}")

    while True:
        try:
            choice = input("\n🔍 Enter the number of the dataset folder to use: ")
            index = int(choice)
            selected = dataset_folders[index]
            break
        except (ValueError, IndexError):
            print("⚠️ Invalid selection. Please try again.")

    dataset_path = Path("dataset") / selected
    print(f"\n✅ Selected dataset: {dataset_path}")
    return dataset_path, selected


dataset_path, DATASET = prompt_for_dataset_folder()

battery_manufacturer = DATASET.split("_")[-1]

# ========== Load battery model from user selection ==========

json_files = sorted(json_path.glob("*.json"))
if not json_files:
    print(f"No .json files found in {json_path}")
    sys.exit(1)  # Changed from exit(1) to sys.exit(1)

choices = [Choice(name=file.name, value=file) for file in json_files]

selected_file = inquirer.select(
    message="Choose a battery model to load:",
    choices=choices,
).execute()

model_hash = selected_file.stem
file_path = selected_file

print(f"Selected model hash: {model_hash}")
print(f"Selected file path: {file_path}")

battery_model = load_battery_model_from_hash(battery_manufacturer, model_hash, json_path)

print(f"Loaded battery model with hash: {battery_model.model_hash}")

# Compose output directory path
output_dir = Path("regression_results") / f"{model_hash}"
output_dir.mkdir(parents=True, exist_ok=True)

# ==========Battery simulation and analysis==========

lin_space = np.linspace(0, 1, 100)
ocv_curves = battery_model.battery_model_data["ocv_curves"]
r_int_curve_params = battery_model.battery_model_data["r_int"]

sorted_items = sorted(
    ((float(temp_str), data) for temp_str, data in ocv_curves.items()),
    key=lambda x: x[0],
)

fig, ax = plt.subplots()
fig.set_size_inches(6, 4)
fig.set_dpi(300)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 6

for temp, ocv_data in sorted_items:
    ocv_data = ocv_curves[str(temp)]
    if DEBUG:
        print(f"OCV data for temperature: {temp}: {ocv_data}")
    ax.plot(
        lin_space,
        rational_linear_rational(lin_space, *ocv_data["ocv_discharge_nc"]),
        label=rf"{temp}_discharge $^\circ$C",
        linewidth=1,
    )
    ax.plot(
        lin_space,
        rational_linear_rational(lin_space, *ocv_data["ocv_charge_nc"]),
        label=rf"{temp}_charge $^\circ$C",
        linewidth=1,
    )

ax.set_title(
    "SOC/VOC curves (scaled from 3.0 - 3.6 terminal voltage, sampled with 100 points)",
    wrap=True,
)
ax.legend()
ax.tick_params(axis="both", which="major", labelsize=6)
ax.set_xlim(xmin=0, xmax=1)
ax.set_xlabel("SoC", fontsize=6)
ax.set_ylabel("Voc [V]", fontsize=6)

export_dir_path = Path("images") / DATASET
export_dir_path.mkdir(parents=True, exist_ok=True)

fig.savefig(export_dir_path / f"{DATASET}_soc_voc_curves.pdf")
fig.savefig(export_dir_path / f"{DATASET}_soc_voc_curves.png")

fig, ax = plt.subplots()
fig.set_size_inches(6, 4)
fig.set_dpi(300)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 6
ax.set_title("Internal resistance evolution", wrap=True)

r_in = []
temp = []

for t in sorted(ocv_curves.keys()):
    r_in.append(ocv_curves[t]["r_int_mean"])
    temp.append(float(t))

r_in = np.array(r_in)
temp = np.array(temp)

ax.plot(temp, r_in, marker="+", label="r_int_mean")
ax.plot(temp, rational_fit(temp, *r_int_curve_params), label="fit", linewidth=1)
ax.tick_params(axis="both", which="major", labelsize=6)
ax.set_xlabel(r"Temperature [$^\circ$C]", fontsize=6)
ax.set_ylabel(r"Rin [$\Omega$]", fontsize=6)

fig.savefig(export_dir_path / f"{DATASET}_R_in_evolution.pdf")
fig.savefig(export_dir_path / f"{DATASET}_R_in_evolution.png")

ukf_model = fuel_gauge(
    ocv_curves,
    model_type="ukf",
    Q=0.001,
    R=1000,
    n=1,
    alpha=1.0,
    beta=1.0,
    kappa=0,
    P_init=0.001,
)
ukf_dynamic_model = fuel_gauge(
    ocv_curves,
    model_type="ukf",
    Q=0.001,
    R=100,
    n=1,
    alpha=2.0,
    beta=2.0,
    kappa=0,
    P_init=0.001,
)
ekf_model = fuel_gauge(ocv_curves, model_type="ekf", Q=0.05, R=20, P_init=100)
ekf_model_adaptive_ref = fuel_gauge(
    ocv_curves,
    model_type="ekf_adaptive",
    Q=0.001,
    R=3000,
    Q_agressive=0.001,
    R_agressive=3000,
    P_init=0.1,
)
ekf_model_adaptive = fuel_gauge(
    ocv_curves,
    model_type="ekf_adaptive",
    Q=0.001,
    R=3000,
    Q_agressive=0.001,
    R_agressive=3000,
    P_init=0.1,
)

simulation_data = list(dataset_path.rglob("**.random_wonder.*.csv"))

print(f"Found {len(simulation_data)} simulation files in {dataset_path}")

try:
    coulomb_counter_est = CoulombCounterEstimator(battery_model=battery_model)
    dummy_est = DummyEstimator(battery_model=battery_model)
    ekf_est = EkfEstimator(
        battery_model=battery_model,
        R=2000,
        Q=0.001,
        Q_agressive=0.001,
        R_agressive=1000,
        P_init=0.1,
    )
    ekf_est2 = EkfEstimator(
        battery_model=battery_model,
        R=2000,
        Q=0.001,
        Q_agressive=0.001,
        R_agressive=1000,
        P_init=0.1,
    )
except Exception as e:
    print(f"Error initializing estimators: {e}")
    sys.exit(1)

for sim in simulation_data:
    print(sim)
    try:
        discharge_profile = load_measured_data(sim)  # Function now matches import alias
    except Exception as e:
        print(f"Error loading data from {sim}: {e}")
        continue

    filtered_profile = copy.copy(discharge_profile)
    filter_length = 80
    filtered_profile.vbat = low_pass_ma_filter(discharge_profile.vbat, filter_length)
    filtered_profile.ibat = low_pass_ma_filter(discharge_profile.ibat, filter_length)

    # Ensure discharge_profile.time is always an array for safe indexing
    time_array = discharge_profile.time
    if np.isscalar(time_array):
        time_array = np.array([time_array])

    sp = int(len(time_array) * 0.25)

    cc_soc, _, cc_sim_time, cc_sim_start, cc_sim_end = run_battery_simulation(
        discharge_profile, coulomb_counter_est
    )
    dummy_soc, _, dummy_sim_time, dummy_sim_start, dummy_sim_end = (
        run_battery_simulation(discharge_profile, dummy_est)
    )
    ekf_soc, _, ekf_sim_time, ekf_sim_start, ekf_sim_end = run_battery_simulation(
        discharge_profile, ekf_est
    )
    ekf_soc2, _, ekf_sim_time2, ekf_sim_start2, ekf_sim_end2 = run_battery_simulation(
        discharge_profile, ekf_est2, sim_start_idx=sp
    )

    voltage_threshold = 3.0
    UVLO = 0  # Initialize UVLO to avoid reference before assignment error
    for i in reversed(range(len(discharge_profile.vbat))):
        if discharge_profile.vbat[i] > voltage_threshold:
            UVLO = i
            break

    fig, ax = plt.subplots(4, 1)
    fig.set_size_inches(10, 10)
    fig.suptitle(f"{sim.name}")

    ax[0].plot(time_to_minutes(time_array), discharge_profile.vbat, label="Vbat")
    ax[0].plot(
        time_to_minutes(time_array), filtered_profile.vbat, label="Vbat_filtered"
    )
    ax[0].vlines(
        time_to_minutes(time_array[UVLO]), *ax[0].get_ylim(), color="red", label="UVLO"
    )
    ax[0].set_title("Battery voltage")
    ax[0].legend()

    ax[1].plot(time_to_minutes(time_array), discharge_profile.ibat)
    ax[1].plot(
        time_to_minutes(
            filtered_profile.time
            if not np.isscalar(filtered_profile.time)
            else np.array([filtered_profile.time])
        ),
        filtered_profile.ibat,
    )
    ax[1].set_title("Battery current")

    ax[2].plot(
        time_to_minutes(time_array[cc_sim_start:cc_sim_end]),
        cc_soc[cc_sim_start:cc_sim_end],
        label="Coulomb counter",
    )
    ax[2].plot(
        time_to_minutes(time_array[dummy_sim_start:dummy_sim_end]),
        dummy_soc[dummy_sim_start:dummy_sim_end],
        label="Dummy",
    )
    ax[2].plot(
        time_to_minutes(time_array[ekf_sim_start:ekf_sim_end]),
        ekf_soc[ekf_sim_start:ekf_sim_end],
        label="EKF",
    )
    ax[2].plot(
        time_to_minutes(time_array[ekf_sim_start2:ekf_sim_end2]),
        ekf_soc2[ekf_sim_start2:ekf_sim_end2],
        label="EKF2",
    )
    ax[2].set_title("SoC")
    ax[2].legend()

    for h in [0, 0.25, 0.5, 0.75, 1.0]:
        ax[2].hlines(h, *ax[2].get_xlim(), color="black", alpha=0.3)
    for frac in [0.25, 0.5, 0.75]:
        ax[2].vlines(
            time_to_minutes(time_array[int(UVLO * frac)]),
            0,
            1,
            color="black",
            alpha=0.3,
        )
    ax[2].vlines(time_to_minutes(time_array[UVLO]), 0, 1, color="red", label="UVLO")

    ax[3].plot(time_to_minutes(time_array), discharge_profile.ntc_temp)
    ax[3].set_title("NTC temperature")

    output_file = output_dir / f"{sim.name}.pdf"

    plt.savefig(output_file)
