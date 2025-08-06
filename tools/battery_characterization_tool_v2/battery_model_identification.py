from pathlib import Path
import numpy as np
import tomllib
import sys
import argparse

import matplotlib.pyplot as plt
from collections import defaultdict
from dataset.battery_profile import cut_charging_phase, cut_discharging_phase
from dataset.battery_dataset import BatteryDataset
from utils.console_formatter import ConsoleFormatter

from battery_model import (
    identify_r_int,
    identify_ocv_curve,
    fit_ocv_curve,
    fit_r_int_curve,
    estimate_r_int,
    estimate_ocv_curve,
)
from generate_c_library import generate_battery_libraries
from fuel_gauge.profile_data_utils import get_mean_temp
from fuel_gauge.battery_model import (
    BatteryModel,
    save_battery_model_to_json,
)

from archive.battery_profiling import rational_fit

DEFAULT_MAX_CHARGE_VOLTAGE = 3.9
DEFAULT_MAX_DISCHARGE_VOLTAGE = 3.0
DEFAULT_OCV_SAMPLES = 100

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Battery characterization tool for processing battery test data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "config_file",
        nargs="?",
        default=None,
        help="Path to TOML configuration file (without .toml extension)",
    )

    parser.add_argument(
        "--debug", action="store_true", help="Enable debug output and plots"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="exported_data",
        help="Output directory for generated data",
    )

    return parser.parse_args()

def prompt_for_config_file():

    config_dir = Path(__file__).parent / "battery_model" / "models"
    toml_files = list(config_dir.glob("*.toml"))

    if not toml_files:
        print("ERROR: No .toml config files found in 'battery_model/models/.'")
        sys.exit(1)

    print("\nAvailable config files in 'battery_model/models/.':")
    for i, file in enumerate(toml_files, 1):
        print(f"  {i}. {file.name}")

    while True:
        try:
            choice = input("Enter config name or number: ").strip()
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(toml_files):
                    return toml_files[index]
                else:
                    print("ERROR: Invalid number.")
            else:
                file_path = config_dir / (
                    choice if choice.endswith(".toml") else f"{choice}.toml"
                )
                if file_path.exists():
                    return file_path
                else:
                    print("ERROR: File not found.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            sys.exit(0)


def load_config(toml_path):
    try:
        with open(toml_path, "rb") as f:
            config = tomllib.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        sys.exit(1)

    # Match exact variable names from config
    dataset_path = Path(config["dataset_path"])
    json_path = Path(config["json_path"])
    battery_manufacturer = config["battery_manufacturer"]
    batteries_to_process = config["batteries_to_process"]
    temperatures_to_process = config["temperatures_to_process"]

    return (
        dataset_path,
        json_path,
        battery_manufacturer,
        batteries_to_process,
        temperatures_to_process,
    )

def create_battery_dataset(dataset_path) -> BatteryDataset:

    battery_dataset = BatteryDataset(dataset_path, load_data=True)
    battery_dataset.print_structure(max_depth=3)

    return battery_dataset

def run_r_int_identification(dataset, debug=False):
    """Takes all switching discharge profiles from the dataset and for every profile estimates the internal
    resistance. All estimations are then fitted with a rational function and its parametrs are returned.
    """
    console = ConsoleFormatter()
    console.subsection("Internal Resistance (R_int) Identification")

    r_int_points = []
    profiles = list(dataset.get_data_list(
        battery_ids=None,
        temperatures=None,
        battery_modes=["switching"],
        mode_phases=["discharging"],
    ))

    total_profiles = len(profiles)
    console.info(f"Found {total_profiles} switching discharge profiles to process")

    # Get all switching discharge profiles across all batteries and temperatures
    for idx, ld_profile in enumerate(profiles, 1):
        profile_data = ld_profile["data"]

        # Progress indicator
        console.progress(
            f"Processing: {ld_profile['battery_id']}.{ld_profile['timestamp_id']}.{ld_profile['temperature']}°C",
            step=idx,
            total=total_profiles
        )

        # Estimate internal resistance on the switching discharge profile
        r_int_estim = identify_r_int(
            profile_data.time,
            profile_data.battery_current,
            profile_data.battery_voltage,
            profile_data.battery_temp,
            debug=debug,
            test_description=f"{ld_profile['battery_id']} {ld_profile['timestamp_id']} {ld_profile['temperature']}°C discharge",
        )

        # extract mean temperature from the analyzed profile
        mean_temp, _ = get_mean_temp(profile_data.battery_temp)
        r_int_points.append([mean_temp, r_int_estim])

    # Fit rational function to the collected data points
    r_int_vector = np.transpose(np.array(r_int_points))
    r_int_rf_params, _ = fit_r_int_curve(r_int_vector[1], r_int_vector[0], debug)

    # Display results
    console.success(f"Collected {len(r_int_points)} R_int estimations")
    console.info(f"Fitted rational function parameters: {r_int_rf_params}")

    return r_int_rf_params

def run_ocv_identification(dataset, r_int_rf_params, charging=False, debug=False):
    """Takes all linear discharge profiles from the dataset and for every profile extracts the open-circuit voltage"""
    console = ConsoleFormatter()
    mode_name = "Charging" if charging else "Discharging"
    console.subsection(f"OCV Identification - {mode_name} Profiles")

    ocv_ident_data = {}

    # Get all linear discharge/charge profiles from the dataset across all batteries and temperatures
    profiles = list(dataset.get_data_list(
        battery_ids=None,
        temperatures=None,
        battery_modes=["linear"],
        mode_phases=["discharging"] if not charging else ["charging"],
    ))

    total_profiles = len(profiles)
    console.info(f"Found {total_profiles} linear {mode_name.lower()} profiles to process")

    for idx, ld_profile in enumerate(profiles, 1):
        # Progress indicator
        console.progress(
            f"{ld_profile['battery_id']}.{ld_profile['timestamp_id']}.{ld_profile['temperature']}°C",
            step=idx,
            total=total_profiles
        )

        # Some of the dataset contain tails from relaxation phase, cut them off`
        if charging:
            profile_data = cut_charging_phase(ld_profile["data"])
        else:
            profile_data = cut_discharging_phase(ld_profile["data"])

        # extract internal resistance from the r_int curve
        real_bat_temp, _ = get_mean_temp(profile_data.battery_temp)
        r_int = estimate_r_int(real_bat_temp, r_int_rf_params)

        # Extract open-circuit voltage (OCV) curve from the linear discharge and charge profiles
        ocv_curve_discharge, total_capacity, effective_capacity = identify_ocv_curve(
            profile_data.time,
            profile_data.battery_voltage,
            profile_data.battery_current,
            r_int,
            max_curve_v=DEFAULT_MAX_CHARGE_VOLTAGE,
            min_curve_v=DEFAULT_MAX_DISCHARGE_VOLTAGE,
            num_of_samples=DEFAULT_OCV_SAMPLES,
            debug=debug,
            test_description=f"{ld_profile['battery_id']} {ld_profile['timestamp_id']} {ld_profile['temperature']}°C {mode_name.lower()}",
        )

        temp_key = ld_profile["temperature"]
        ocv_ident_data.setdefault(
            temp_key,
            {
                "ocv_curve_points": [],
                "total_capacity": [],
                "effective_capacity": [],
                "real_bat_temp": [],
            },
        )

        ocv_ident_data[temp_key]["ocv_curve_points"].append(ocv_curve_discharge)
        ocv_ident_data[temp_key]["total_capacity"].append(total_capacity)
        ocv_ident_data[temp_key]["effective_capacity"].append(effective_capacity)
        ocv_ident_data[temp_key]["real_bat_temp"].append(real_bat_temp)

    console.success(f"Processed {total_profiles} {mode_name.lower()} profiles")
    return ocv_ident_data


def extract_soc_and_rint_curves(
    dataset,
    filter_batteries,
    characterized_temperatures_deg,
    debug=False,
):

    filtered_dataset = dataset.filter(
        battery_ids=filter_batteries,
        temperatures=characterized_temperatures_deg,
        battery_modes=["switching", "linear"],
    )

    r_int_rf_params = run_r_int_identification(filtered_dataset, debug=debug)

    ocv_data_discharge = run_ocv_identification(
        filtered_dataset,
        r_int_rf_params,
        charging=False,
        debug=debug,
    )

    ocv_data_charge = run_ocv_identification(
        filtered_dataset,
        r_int_rf_params,
        charging=True,
        debug=debug,
    )

    """
    All ocv data for charge and discharge profiles are ready, fit the ocv
    curves and assign them with real battery temperatures, then store them
    in ocv_curves dict.
    """
    ocv_curves = {}

    for temp in characterized_temperatures_deg:

        ocv_profiles_discharge = np.array(
            [
                np.concatenate(
                    [d[0] for d in ocv_data_discharge[temp]["ocv_curve_points"]]
                ),  # X values concatenated
                np.concatenate(
                    [d[1] for d in ocv_data_discharge[temp]["ocv_curve_points"]]
                ),  # Y values concatenated
            ]
        )

        dsg_ocv_params, dsg_ocv_params_complete = fit_ocv_curve(ocv_profiles_discharge)
        dsg_temp = np.mean(ocv_data_discharge[temp]["real_bat_temp"])
        dsg_ef_cap = np.mean(ocv_data_discharge[temp]["effective_capacity"])
        dsg_total_cap = np.mean(ocv_data_discharge[temp]["total_capacity"])

        ocv_profiles_charge = np.array(
            [
                np.concatenate(
                    [d[0] for d in ocv_data_charge[temp]["ocv_curve_points"]]
                ),  # X values concatenated
                np.concatenate(
                    [d[1] for d in ocv_data_charge[temp]["ocv_curve_points"]]
                ),  # Y values concatenated
            ]
        )

        chg_ocv_params, chg_ocv_params_complete = fit_ocv_curve(ocv_profiles_charge)
        chg_temp = np.mean(ocv_data_charge[temp]["real_bat_temp"])
        chg_ef_cap = np.mean(ocv_data_charge[temp]["effective_capacity"])
        chg_total_cap = np.mean(ocv_data_charge[temp]["total_capacity"])

        ocv_curves[temp] = {
            "ocv_dischg": dsg_ocv_params_complete,
            "ocv_dischg_nc": dsg_ocv_params,
            "bat_temp_dischg": dsg_temp,
            "total_capacity_dischg": dsg_total_cap,
            "effective_capacity_dischg": dsg_ef_cap,
            "ocv_chg": chg_ocv_params_complete,
            "ocv_chg_nc": chg_ocv_params,
            "bat_temp_chg": chg_temp,
            "total_capacity_chg": chg_total_cap,
            "effective_capacity_chg": chg_ef_cap,
        }

    if(True):

        fig, ax = plt.subplots(2,1)
        for ocv_curve in ocv_curves.values():

            soc_axis = np.linspace(0,1,100)

            ax[0].plot(soc_axis, estimate_ocv_curve(soc_axis, ocv_curve["ocv_dischg_nc"]), label=f"Discharge {ocv_curve['bat_temp_dischg']}°C")
            ax[0].plot(soc_axis, estimate_ocv_curve(soc_axis, ocv_curve["ocv_chg_nc"]), label=f"Charge {ocv_curve['bat_temp_chg']}°C")

            ax[1].plot(ocv_curve["bat_temp_dischg"], ocv_curve["total_capacity_dischg"], 'o', label=f"Total Capacity Discharge {ocv_curve['bat_temp_dischg']}°C")
            ax[1].plot(ocv_curve["bat_temp_chg"], ocv_curve["total_capacity_chg"], 'o', label=f"Total Capacity Charge {ocv_curve['bat_temp_chg']}°C")

        ax[0].set_title("OCV Curves")
        ax[0].set_xlabel("SoC")
        ax[0].set_ylabel("Voltage (V)")
        ax[0].legend()
        ax[1].set_title("Total Capacity")
        ax[1].set_xlabel("Temperature (°C)")
        ax[1].set_ylabel("Capacity (Ah)")
        ax[1].legend()

        plt.tight_layout()

    return ocv_curves, r_int_rf_params

def main():

    # Initialize console formatter
    console = ConsoleFormatter()

    # Print header
    console.header(
        "BATTERY CHARACTERIZATION TOOL",
        "Advanced Battery Model Identification and Analysis",
        width=85
    )

    # Parse command line arguments
    args = parse_arguments()

    # Handle config file selection
    if args.config_file:
        config_file = Path(args.config_file)

        if not config_file.suffix == ".toml":
            raise IOError("Config file must be a .toml file")

        if not config_file_path.exists():
            print(f"Config file '{config_file}' not found.")
            sys.exit(1)
    else:
        config_file_path = prompt_for_config_file()

    # Load config
    console.section("CONFIGURATION SETUP")

    (
        dataset_path,
        json_path,
        battery_manufacturer,
        batteries_to_process,
        temperatures_to_process,
    ) = load_config(config_file_path)

    # Display configuration information professionally
    config_data = {
        "Dataset Path": str(dataset_path),
        "Manufacturer": battery_manufacturer,
        "Temperatures": f"{temperatures_to_process}°C",
        "Output Directory": args.output_dir,
        "Debug Mode": "Enabled" if args.debug else "Disabled"
    }

    console.key_value_pairs(config_data, "Configuration Summary")
    console.success("Configuration loaded successfully")

    # Dataset loading section
    console.section("DATASET LOADING")
    console.info("Loading battery dataset...")
    dataset = create_battery_dataset(dataset_path)
    console.success(f"Dataset loaded: {dataset}")

    # Main processing section
    console.section("BATTERY MODEL IDENTIFICATION")
    console.info("Starting battery characterization analysis...")

    ocv_curves, r_int_rf_params = (
        extract_soc_and_rint_curves(
            dataset,
            filter_batteries=batteries_to_process,
            characterized_temperatures_deg=temperatures_to_process,
            debug=args.debug,
        )
    )

    console.success("Battery model identification completed")

    console.section("RESULTS SUMMARY")

    # Create a sample results table
    headers = ["Temperature", "Discharge capacity", "Charge capacity", "R_int"]

    rows = []
    for temp, data in sorted(ocv_curves.items(), key=lambda x: x[0]):
        rows.append([
            f"{temp}°C",
            f"{data['total_capacity_dischg']*1000:.2f} mAh",
            f"{data['total_capacity_chg']*1000:.2f} mAh",
            f"{estimate_r_int(float(temp), r_int_rf_params)*1000:.2f} mΩ"
        ])

    console.table(headers, rows, title="Battery Model Analysis Results")

    console.success("Analysis results table generated")

    # Prepare battery model data
    battery_model_data = {
        "r_int": r_int_rf_params,
        "ocv_curves": ocv_curves,
        "battery_vendor": battery_manufacturer,
    }

    battery_model = BatteryModel(battery_model_data, dataset.get_dataset_hash())

    generate_battery_libraries(
        battery_model_data,
        output_dir=args.output_dir,
        battery_name=battery_manufacturer,
    )

    save_battery_model_to_json(battery_model, json_path)

    # Show completion summary
    console.footer()

    if args.debug:
        # Show the plots if in debug mode
        plt.show()

if __name__ == "__main__":
    main()
