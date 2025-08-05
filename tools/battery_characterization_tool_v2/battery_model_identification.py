from pathlib import Path
import numpy as np
import tomllib
import sys
import argparse

import matplotlib.pyplot as plt
from collections import defaultdict
from dataset.data_filtering import filter_dataset_constants, count_entries
from dataset.battery_profile import cut_charging_phase, cut_discharging_phase
from dataset.battery_dataset import BatteryDataset

from battery_model import (
    identify_r_int,
    identify_ocv_curve,
    fit_ocv_curve,
    fit_r_int_curve,
    estimate_r_int,
    estimate_ocv_curve,
)
from generate_battery_libraries_v2 import generate_battery_libraries
from dataset.battery_profile import load_battery_profile
from fuel_gauge.profile_data_utils import get_mean_temp
from fuel_gauge.battery_model import (
    BatteryModel,
    hashlib,
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

    print(f"Dataset loaded: {battery_dataset}")

    battery_dataset.print_structure(max_depth=3)

    return battery_dataset


def run_r_int_identification(dataset, debug=False):
    """Takes all switching discharge profiles from the dataset and for every profile estimates the internal
    resistance. All estimations are then fitted with a rational function and its parametrs are returned.
    """

    r_int_points = []

    # Get all switching discharge profiles across all batteries and temperatures
    for ld_profile in dataset.get_data_list(
        battery_ids=None,
        temperatures=None,
        battery_modes=["switching"],
        mode_phases=["discharging"],
    ):

        print(
            f"Processing profile: {ld_profile['battery_id']}.{ld_profile['timestamp_id']}.{ld_profile['battery_mode']}.{ld_profile['mode_phase']}.{ld_profile['temperature']}"
        )

        profile_data = ld_profile["data"]

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

    # r_int_points_sorted = sorted(r_int_points, key=lambda x: x[0])
    r_int_vector = np.transpose(np.array(r_int_points))
    r_int_rf_params, _ = fit_r_int_curve(r_int_vector[1], r_int_vector[0], debug)

    return r_int_rf_params


def run_ocv_identification(dataset, r_int_rf_params, charging=False, debug=False):
    """Takes all linear discharge profiles from the dataset and for every profile extracts the open-circuit voltage"""
    ocv_ident_data = {}

    # Get all linear discharge/charge profiles from the dataset across all batteries and temperatures
    for ld_profile in dataset.get_data_list(
        battery_ids=None,
        temperatures=None,
        battery_modes=["linear"],
        mode_phases=["discharging"] if not charging else ["charging"],
    ):

        print(
            f"Processing profile: {ld_profile['battery_id']}.{ld_profile['timestamp_id']}.{ld_profile['battery_mode']}.{ld_profile['mode_phase']}.{ld_profile['temperature']}"
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
            test_description=f"{ld_profile['battery_id']} {ld_profile['timestamp_id']} {ld_profile['temperature']}°C discharge",
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

    return ocv_ident_data


def extract_soc_and_rint_curves(
    dataset,
    filter_batteries,
    characterized_temperatures_deg,
    debug=False,
):

    # def hash_leaf_file_names(d):
    #     leaf_keys = []

    #     def collect_leaves(node):
    #         if isinstance(node, dict):
    #             for key, value in node.items():
    #                 if isinstance(value, dict) and value:
    #                     collect_leaves(value)
    #                 elif not isinstance(value, dict):
    #                     pass  # skip — we only want keys (file names)
    #                 else:
    #                     leaf_keys.append(key)  # file name as dict key

    #     collect_leaves(d)

    #     # Sort keys for deterministic hash
    #     leaf_keys_sorted = sorted(leaf_keys)
    #     joined_keys = "\n".join(leaf_keys_sorted)

    #     return hashlib.sha256(joined_keys.encode("utf-8")).hexdigest()

    # # Filter dataset to relevant components
    # filtered_dataset = filter_dataset_constants(
    #     dataset,
    #     filtered_batteries=filter_batteries,
    #     filtered_modes=["switching", "linear"],
    #     filtered_phases=["charging", "discharging"],
    #     filtered_temps=characterized_temperatures_deg,
    # )

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

        ocv_curves[round(temp, 2)] = {
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

    plt.show()
    sys.exit(0)

    for temp in characterized_temperatures_deg:

        print(f"\nProcessing temperature: {temp}°C")
        profiles = []

        # Sweep all batteries in the dataset for given temperature
        filtered_temp = filtered_dataset.filter(temperatures=[temp])

        for battery in filtered_temp.get_battery_ids():

            # Identify switching profiles to identify internal resistance
            for i in range(1, 4):
                try:
                    switching_discharge = filtered_temp.get_data(
                        battery, temp, "switching", "discharging", timestamp
                    )
                    switching_charge = filtered_temp.get_data(
                        battery, temp, "switching", "charging", timestamp
                    )
                    linear_discharge = filtered_temp.get_data(
                        battery, temp, "linear", "discharging", timestamp
                    )
                    linear_charge = filtered_temp.get_data(
                        battery, temp, "linear", "charging", timestamp
                    )
                except:
                    print(
                        f"Error, some of the profiles are missing for battery {battery}, temp {temp}, timestamp {timestamp}"
                    )
                    continue

                # Make sure the profile do not have any tails from relaxation phase
                switching_discharge = cut_discharging_phase(switching_discharge)
                switching_charge = cut_charging_phase(switching_charge)
                linear_charge = cut_charging_phase(linear_charge)
                linear_discharge = cut_discharging_phase(linear_discharge)

                # Estimate internal resistance on the switching discharge profile
                r_int_estim = identify_r_int(
                    switching_discharge.time,
                    switching_discharge.battery_current,
                    switching_discharge.battery_voltage,
                    switching_discharge.battery_temp,
                    debug=debug,
                    test_description=f"{battery} {temp}°C discharge",
                )

                """
                Internal resistance estimated on the charging waveform gives questionable results,
                so its not used for anything right now, we keep it here for completeness and use the
                estimated resistance on discharge waveforms.
                """
                r_int_estim_charge = identify_r_int(
                    switching_charge.time,
                    switching_charge.battery_current,
                    switching_charge.battery_voltage,
                    switching_charge.battery_temp,
                    debug=debug,
                    test_description=f"{battery} {temp}°C charge",
                )

                # Extract open-circuit voltage (OCV) curve from the linear discharge and charge profiles
                ocv_curve_discharge, total_capacity, effective_capacity = (
                    identify_ocv_curve(
                        linear_discharge.time,
                        linear_discharge.battery_voltage,
                        linear_discharge.battery_current,
                        r_int_estim,
                        max_curve_v=DEFAULT_MAX_CHARGE_VOLTAGE,
                        min_curve_v=DEFAULT_MAX_DISCHARGE_VOLTAGE,
                        num_of_samples=DEFAULT_OCV_SAMPLES,
                        debug=debug,
                        test_description=f"{battery} {temp}°C discharge",
                    )
                )

                ocv_curve_charge, _, effective_capacity_charge = identify_ocv_curve(
                    linear_charge.time,
                    linear_charge.battery_voltage,
                    linear_charge.battery_current,
                    r_int_estim,
                    max_curve_v=DEFAULT_MAX_CHARGE_VOLTAGE,
                    min_curve_v=DEFAULT_MAX_DISCHARGE_VOLTAGE,
                    num_of_samples=DEFAULT_OCV_SAMPLES,
                    debug=debug,
                    test_description=f"{battery} {temp}°C charge",
                )

                mean_temp_discharge, _ = get_mean_temp(switching_discharge.battery_temp)
                r_int_points.append([mean_temp_discharge, r_int_estim])

                mean_temp_charge, _ = get_mean_temp(switching_charge.battery_temp)
                r_int_points_charge.append([mean_temp_charge, r_int_estim_charge])

                entry = {
                    "data": linear_discharge,
                    "chamber_temp": temp,
                    "ntc_temp": mean_temp_discharge,
                    "r_int": r_int_estim,
                    "r_int_charge": r_int_estim_charge,
                    "max_chg_voltage": DEFAULT_MAX_CHARGE_VOLTAGE,
                    "max_disch_voltage": DEFAULT_MAX_DISCHARGE_VOLTAGE,
                    "ocv_curve": ocv_curve_discharge,
                    "ocv_curve_charge": ocv_curve_charge,
                    "total_capacity": effective_capacity,
                    "total_capacity_charge": effective_capacity_charge,
                    "capacity_yield": total_capacity - effective_capacity,
                }

                profiles.append(entry)

        if not profiles:
            print(f"ERROR: No usable profiles at {temp}°C")
            continue

        # Prepare ocv discharge profiles for concatenation and fitting
        ocv_profiles_discharge = np.array(
            [
                np.concatenate(
                    [p["ocv_curve"][0] for p in profiles]
                ),  # X values concatenated
                np.concatenate(
                    [p["ocv_curve"][1] for p in profiles]
                ),  # Y values concatenated
            ]
        )

        # Prepare ocv charge profiles similarly
        ocv_profiles_charge = np.array(
            [
                np.concatenate([p["ocv_curve_charge"][0] for p in profiles]),
                np.concatenate([p["ocv_curve_charge"][1] for p in profiles]),
            ]
        )

        # Fit ocv curves on the concatenated data (discharge and charge)
        curve_params, curve_params_complete = fit_ocv_curve(ocv_profiles_discharge)
        curve_params_charge, curve_params_charge_complete = fit_ocv_curve(
            ocv_profiles_charge
        )

        """
        # Extract arrays of all ocv curve Y-values (discharge and charge) for averaging & std
        #ocv_discharge_y = np.array([p["SoC_curve"][1] for p in profiles])
        #ocv_charge_y = np.array([p["SoC_curve_charge"][1] for p in profiles])
        # Average and std deviation of ocv discharge curve (pointwise)
        #ocv_mean_y = np.mean(ocv_discharge_y, axis=0)
        #ocv_std_y = np.std(ocv_discharge_y, axis=0)
        ocv_mean = np.array(
            [profiles[0]["SoC_curve"][0], ocv_mean_y]
        )  # Assuming all x same
        """
        # Mean and std dev for NTC temperature
        ntc_temps = np.array([p["ntc_temp"] for p in profiles])
        ntc_temp_mean = np.mean(ntc_temps)
        ntc_temp_std = np.std(ntc_temps)

        # Aggregate capacity and resistance stats with std dev
        total_capacity_arr = np.array([p["total_capacity"] for p in profiles])
        total_capacity_charge_arr = np.array(
            [p["total_capacity_charge"] for p in profiles]
        )
        # capacity_yield_arr = np.array([p["capacity_yield"] for p in profiles])
        r_int_arr = np.array([p["r_int"] for p in profiles])

        total_capacity_mean = np.mean(total_capacity_arr)
        total_capacity_std = np.std(total_capacity_arr)

        total_capacity_charge_mean = np.mean(total_capacity_charge_arr)
        total_capacity_charge_std = np.std(total_capacity_charge_arr)

        # capacity_yield_mean = np.mean(capacity_yield_arr)
        # capacity_yield_std = np.std(capacity_yield_arr)

        r_int_mean = np.mean(r_int_arr)
        r_int_std = np.std(r_int_arr)

        # Store results in dictionaries indexed by rounded ntc temp mean
        ocv_curves[round(ntc_temp_mean, 2)] = {
            "ocv_discharge": curve_params_complete,
            "ocv_charge": curve_params_charge_complete,
            "ocv_discharge_nc": curve_params,
            "ocv_charge_nc": curve_params_charge,
            "total_capacity_discharge_mean": total_capacity_mean,
            "total_capacity_discharge_std": total_capacity_std,
            "total_capacity_charge_mean": total_capacity_charge_mean,
            "total_capacity_charge_std": total_capacity_charge_std,
            "r_int_mean": r_int_mean,
            "r_int_std": r_int_std,
            "ntc_temp_mean": ntc_temp_mean,
            "ntc_temp_std": ntc_temp_std,
        }

    # ========Testing and printing functions========
    if debug:

        def fmt_float(val, precision=4):
            if isinstance(val, (int, float)):
                return f"{val:.{precision}f}"
            return str(val)

        print("=== OCV Curves ===")
        for temp, data in ocv_curves.items():
            print(f"Temperature: {temp}°C")
            print(f"  - OCV Discharge curve params: {data['ocv_discharge']}")
            print(f"  - OCV Charge curve params: {data['ocv_charge']}")
            print(
                f"  - Total Capacity Discharge: mean={fmt_float(data.get('total_capacity_discharge_mean'))}, + std={fmt_float(data.get('total_capacity_discharge_std'))}"
            )
            print(
                f"  - Total Capacity Charge: mean={fmt_float(data.get('total_capacity_charge_mean'))}, std={fmt_float(data.get('total_capacity_charge_std'))}"
            )
            print(
                f"  - R_int: mean={fmt_float(data.get('r_int_mean'), 6)}, std={fmt_float(data.get('r_int_std'), 6)}"
            )
            print(
                f"  - NTC Temp: mean={fmt_float(data.get('ntc_temp_mean'), 2)}, std={fmt_float(data.get('ntc_temp_std'), 2)}"
            )
            print()

        print("=== R_int Discharge ===")
        if isinstance(r_int_points, list):
            print(f"R_int points count: {len(r_int_points)}")
        else:
            print(r_int_points)
        print()

        print("=== R_int Charge ===")
        if isinstance(r_int_points_charge, list):
            print(f"R_int_charge points count: {len(r_int_points_charge)}")
        else:
            print(r_int_points_charge)
        print()

    return ocv_curves, r_int_points, r_int_points_charge, file_name_hash


def main():
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
    (
        dataset_path,
        json_path,
        battery_manufacturer,
        batteries_to_process,
        temperatures_to_process,
    ) = load_config(config_file_path)

    print("SUCCESS: Configuration Loaded:")
    print(f"  Dataset path: {dataset_path}")
    print(f"  Manufacturer: {battery_manufacturer}")
    print(f"  Temperatures: {temperatures_to_process}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Debug mode: {args.debug}")

    dataset = create_battery_dataset(dataset_path)

    ocv_curves, rint_points, rint_points_charge, file_name_hash = (
        extract_soc_and_rint_curves(
            dataset,
            filter_batteries=batteries_to_process,
            characterized_temperatures_deg=temperatures_to_process,
            debug=args.debug,
        )
    )  # Sort rint_points by temperature (assumed to be first element of each sublist)
    rint_points_sorted = sorted(rint_points, key=lambda x: x[0])
    r_int_vector = np.transpose(np.array(rint_points_sorted))
    r_int_curve_params, _ = fit_r_int_curve(
        r_int_vector[1], r_int_vector[0], args.debug
    )
    """
    # Same for charge
    rint_points_charge_sorted = sorted(rint_points_charge, key=lambda x: x[0])
    r_int_vector_charge = np.transpose(np.array(rint_points_charge_sorted))
    r_int_curve_params_charge = fit_R_int_curve(r_int_vector_charge[1], r_int_vector_charge[0])
    """
    # Plot R_int curves
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 6

    # Sorting only for plotting lines (to avoid zigzag)
    sorted_discharge = sorted(zip(r_int_vector[0], r_int_vector[1]))
    x_d, y_d = zip(*sorted_discharge)
    """
    sorted_charge = sorted(zip(r_int_vector_charge[0], r_int_vector_charge[1]))
    x_c, y_c = zip(*sorted_charge)
    """
    # Compute fitted values on sorted x for smooth curves
    fit_y_d = rational_fit(np.array(x_d), *r_int_curve_params)
    # fit_y_c = rational_fit(np.array(x_c), *r_int_curve_params_charge)

    # Auto Y scaling considering all data and fits
    # all_rints = np.concatenate([y_d, y_c, fit_y_d, fit_y_c])
    ymin = np.min(y_d)
    ymax = np.max(fit_y_d)
    yrange = ymax - ymin
    margin = 0.1 * yrange if yrange > 0 else 0.01

    # Plot
    ax.set_title(f"Rint curve params {r_int_curve_params}", wrap=True)
    ax.set_xlabel(r"Temperature [$^\circ$C]", fontsize=6)
    ax.set_ylabel(r"Internal Resistance $R_{int}$ [$\Omega$]", fontsize=6)

    ax.plot(x_d, y_d, marker="+", linestyle="", label="Rint estimation (discharge)")
    # ax.plot(x_c, y_c, marker='+', linestyle='', label="Rint estimation (charge)")
    ax.plot(x_d, fit_y_d, label="Rint curve fit (discharge)")
    # ax.plot(x_c, fit_y_c, label="Rint curve fit (charge)")

    ax.legend()
    ax.tick_params(axis="both", which="major", labelsize=6)
    ax.set_xlim(xmin=0, xmax=50)
    ax.set_ylim(ymin - margin, ymax + margin)

    print(f"Rint curve params {r_int_curve_params}")

    # Prepare battery model data
    battery_model_data = {
        "r_int": r_int_curve_params,
        "ocv_curves": ocv_curves,
        "battery_vendor": battery_manufacturer,
    }

    battery_model = BatteryModel(battery_model_data, file_name_hash)

    generate_battery_libraries(
        battery_model_data,
        output_dir=args.output_dir,
        battery_name=battery_manufacturer,
    )

    save_battery_model_to_json(battery_model, json_path)

    if args.debug:
        # Show the plots if in debug mode
        plt.show()


if __name__ == "__main__":
    main()
