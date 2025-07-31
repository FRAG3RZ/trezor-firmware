from pathlib import Path
import numpy as np
import tomllib
import sys
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from utils.data_filtering import filter_dataset_constants, count_entries

from battery_model import identify_r_int, identify_ocv_curve

from fuel_gauge.battery_profiling import (
    estimate_R_int,
    extract_SoC_curve,
    extract_SoC_curve_charging,
    rational_fit,
    fit_soc_curve,
    fit_R_int_curve,
)
from generate_battery_libraries_v2 import generate_battery_libraries
from utils.data_convertor import load_measured_data_new
from fuel_gauge.profile_data_utils import get_mean_temp
from fuel_gauge.battery_model import (
    BatteryModel,
    hashlib,
    save_battery_model_to_json,
)

# ======CONFIG LOADING ========


def prompt_for_config_file():
    config_dir = Path(__file__).parent / "battery_model" / "models"
    toml_files = list(config_dir.glob("*.toml"))

    if not toml_files:
        print("❌ No .toml config files found in 'battery_model/models/.'")
        sys.exit(1)

    print("\n📂 Available config files in 'battery_model/models/.':")
    for i, file in enumerate(toml_files, 1):
        print(f"  {i}. {file.name}")

    while True:
        try:
            choice = input("🔧 Enter config name or number: ").strip()
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(toml_files):
                    return toml_files[index]
                else:
                    print("❌ Invalid number.")
            else:
                file_path = config_dir / (
                    choice if choice.endswith(".toml") else f"{choice}.toml"
                )
                if file_path.exists():
                    return file_path
                else:
                    print("❌ File not found.")
        except KeyboardInterrupt:
            print("\n⛔ Cancelled.")
            sys.exit(0)


def load_config(toml_path):
    try:
        with open(toml_path, "rb") as f:
            config = tomllib.load(f)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        sys.exit(1)

    # Match exact variable names from config
    dataset_path = Path(config["dataset_path"]).glob("*.csv")
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


# ==========Disctionary structure for the data===========


# Helper to safely parse filenames
def extract_file_info(file):
    parts = file.stem.split(".")
    if len(parts) < 5:
        return None
    return {
        "battery": parts[0],
        "timestamp": parts[1],
        "mode": parts[2],
        "phase": parts[3],
        "temp": parts[4],
        "path": file,
    }


# Build the library dataset in the format [battery][time][mode][phase][temp]
def build_library_dataset():
    # Use recursive defaultdicts to build a 5-level nested dict
    library = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )

    for file in dataset_path:
        info = extract_file_info(file)
        if info is None:
            continue

        # Extract metadata
        battery = info["battery"]
        timestamp = info["timestamp"]
        mode = info["mode"]
        phase = info["phase"]
        temp = info["temp"]

        # ⛔ Skip if the phase is 'done'
        if phase == "done":
            print(f"⏭️ Skipping {file.name} — phase is 'done'")
            continue

        try:
            data = load_measured_data_new(info["path"])
            library[battery][timestamp][mode][phase][temp] = data
        except Exception as e:
            print(f"⚠️ Failed to load {file.name}: {e}")

    print(f"Dataset built with {len(library)} batteries.")
    return library


# ========Graphing functions========


def extract_soc_and_rint_curves(
    dataset,
    filter_batteries,
    characterized_temperatures_deg,
    soc_curve_max_chg_voltage,
    soc_curve_max_dchg_voltage,
    soc_curve_points_num,
):

    # ====SET THE DEBUG FLAG FOR EXTRA PRINTS====
    debug = False

    def hash_leaf_file_names(d):
        leaf_keys = []

        def collect_leaves(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if isinstance(value, dict) and value:
                        collect_leaves(value)
                    elif not isinstance(value, dict):
                        pass  # skip — we only want keys (file names)
                    else:
                        leaf_keys.append(key)  # file name as dict key

        collect_leaves(d)

        # Sort keys for deterministic hash
        leaf_keys_sorted = sorted(leaf_keys)
        joined_keys = "\n".join(leaf_keys_sorted)

        return hashlib.sha256(joined_keys.encode("utf-8")).hexdigest()

    # Filter dataset to relevant components
    filtered_dataset = filter_dataset_constants(
        dataset,
        filtered_batteries=filter_batteries,
        filtered_modes=["switching", "linear"],
        filtered_phases=["charging", "discharging"],
        filtered_temps=characterized_temperatures_deg,
    )

    print(
        f"✅ Dataset contains {count_entries(filtered_dataset)} entries after filtering."
    )
    file_name_hash = hash_leaf_file_names(filtered_dataset)
    print(f"File names hash: {file_name_hash}")

    ocv_curves = {}
    r_int_points = []
    r_int_points_charge = []

    def find_profiles_across_timestamps(battery, temp):
        temp_key = str(temp)
        found = {
            "switching_discharge": None,
            "switching_charge": None,
            "linear_discharge": None,
            "linear_charge": None,
        }
        for timestamp in sorted(filtered_dataset[battery]):
            node = filtered_dataset[battery][timestamp]

            found["switching_discharge"] = found["switching_discharge"] or node.get(
                "switching", {}
            ).get("discharging", {}).get(temp_key)
            found["switching_charge"] = found["switching_charge"] or node.get(
                "switching", {}
            ).get("charging", {}).get(temp_key)
            found["linear_discharge"] = found["linear_discharge"] or node.get(
                "linear", {}
            ).get("discharging", {}).get(temp_key)
            found["linear_charge"] = found["linear_charge"] or node.get(
                "linear", {}
            ).get("charging", {}).get(temp_key)

            if all(found.values()):
                break

        return found

    for temp in characterized_temperatures_deg:
        print(f"\n🌡️ Processing temperature: {temp}°C")
        profiles = []

        for battery in sorted(filtered_dataset):
            print(f"🔋 Processing battery: {battery}")
            found = find_profiles_across_timestamps(battery, temp)

            if not all(found.values()):
                print(f"⚠️ Missing profiles for battery {battery}, temp {temp}:")
                for k, v in found.items():
                    if v is None:
                        print(f"  - Missing {k.replace('_', ' ')}")
                continue

            # All profiles are now safe to use
            switching_discharge = found["switching_discharge"]
            switching_charge = found["switching_charge"]
            linear_discharge = found["linear_discharge"]
            linear_charge = found["linear_charge"]

            R_int_estim = identify_r_int(
                switching_discharge.time,
                switching_discharge.battery_current,
                switching_discharge.battery_voltage,
                switching_discharge.battery_temp,
                debug=True,
            )

            # R_int_estim = estimate_R_int(
            #     switching_discharge.time,
            #     switching_discharge.battery_current,
            #     switching_discharge.battery_voltage,
            #     switching_discharge.battery_temp,
            #     debug=True,
            # )

            R_int_estim_charge = estimate_R_int(
                switching_charge.time,
                switching_charge.battery_current,
                switching_charge.battery_voltage,
                switching_charge.battery_temp,
                debug=False,
            )

            SoC_curve, total_capacity, effective_capacity = extract_SoC_curve(
                temp,
                linear_discharge,
                R_int_estim,
                max_chg_voltage=soc_curve_max_chg_voltage,
                max_dischg_voltage=soc_curve_max_dchg_voltage,
                num_of_points=soc_curve_points_num,
                debug=False,
            )

            SoC_curve_charge, total_capacity_charge, effective_capacity_charge = (
                extract_SoC_curve_charging(
                    temp,
                    linear_charge,
                    R_int_estim,
                    max_chg_voltage=3.9,
                    max_dischg_voltage=3.0,
                    num_of_points=soc_curve_points_num,
                    debug=False,
                )
            )

            mean_temp_discharge, _ = get_mean_temp(switching_discharge.battery_temp)
            r_int_points.append([mean_temp_discharge, R_int_estim])

            mean_temp_charge, _ = get_mean_temp(switching_charge.battery_temp)
            r_int_points_charge.append([mean_temp_charge, R_int_estim_charge])

            entry = {
                "data": linear_discharge,
                "ambient_temp": temp,
                "ntc_temp": mean_temp_discharge,
                "R_int": R_int_estim,
                "R_int_charge": R_int_estim_charge,
                "max_chg_voltage": soc_curve_max_chg_voltage,
                "max_disch_voltage": soc_curve_max_dchg_voltage,
                "SoC_curve": SoC_curve,
                "SoC_curve_charge": SoC_curve_charge,
                "total_capacity": effective_capacity,
                "total_capacity_charge": effective_capacity_charge,
                "capacity_yield": total_capacity - effective_capacity,
            }

            profiles.append(entry)

        if not profiles:
            print(f"🚫 No usable profiles at {temp}°C")
            continue

        # Prepare ocv discharge profiles for concatenation and fitting
        ocv_profiles_discharge = np.array(
            [
                np.concatenate(
                    [p["SoC_curve"][0] for p in profiles]
                ),  # X values concatenated
                np.concatenate(
                    [p["SoC_curve"][1] for p in profiles]
                ),  # Y values concatenated
            ]
        )

        # Prepare ocv charge profiles similarly
        ocv_profiles_charge = np.array(
            [
                np.concatenate([p["SoC_curve_charge"][0] for p in profiles]),
                np.concatenate([p["SoC_curve_charge"][1] for p in profiles]),
            ]
        )
        print(ocv_profiles_charge)
        # Fit ocv curves on the concatenated data (discharge and charge)
        curve_params, curve_params_complete = fit_soc_curve(
            ocv_profiles_discharge, "OCV Discharge"
        )
        curve_params_charge, curve_params_charge_complete = fit_soc_curve(
            ocv_profiles_charge, "OCV Charge"
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
        R_int_arr = np.array([p["R_int"] for p in profiles])

        total_capacity_mean = np.mean(total_capacity_arr)
        total_capacity_std = np.std(total_capacity_arr)

        total_capacity_charge_mean = np.mean(total_capacity_charge_arr)
        total_capacity_charge_std = np.std(total_capacity_charge_arr)

        # capacity_yield_mean = np.mean(capacity_yield_arr)
        # capacity_yield_std = np.std(capacity_yield_arr)

        R_int_mean = np.mean(R_int_arr)
        R_int_std = np.std(R_int_arr)

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
            "r_int_mean": R_int_mean,
            "r_int_std": R_int_std,
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


if __name__ == "__main__":

    # Allow: python config_from_toml.py myconfig.toml
    if len(sys.argv) > 1:
        toml_file = sys.argv[1]
        if not toml_file.endswith(".toml"):
            toml_file += ".toml"
        config_file_path = Path(toml_file)
        if not config_file_path.exists():
            print(f"❌ File '{toml_file}' not found.")
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

    print("✅ Configuration Loaded:")
    print(f"  Dataset path: {config_file_path}")
    print(f"  Manufacturer: {battery_manufacturer}")
    print(f"  Temperatures: {temperatures_to_process}")

    dataset = build_library_dataset()

    ocv_curves, rint_points, rint_points_charge, file_name_hash = (
        extract_soc_and_rint_curves(
            dataset,
            filter_batteries=batteries_to_process,
            characterized_temperatures_deg=temperatures_to_process,
            soc_curve_max_chg_voltage=3.6,
            soc_curve_max_dchg_voltage=3.0,
            soc_curve_points_num=100,
        )
    )

    # Sort rint_points by temperature (assumed to be first element of each sublist)
    rint_points_sorted = sorted(rint_points, key=lambda x: x[0])
    r_int_vector = np.transpose(np.array(rint_points_sorted))
    r_int_curve_params = fit_R_int_curve(r_int_vector[1], r_int_vector[0])
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
        output_dir="exported_libraries_updated",
        battery_name=battery_manufacturer,
    )

    save_battery_model_to_json(battery_model, json_path)
