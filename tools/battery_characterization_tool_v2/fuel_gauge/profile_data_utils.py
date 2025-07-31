
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

@dataclass
class BatteryAnalysisData:
    time: np.ndarray
    vbat: np.ndarray
    ibat: np.ndarray
    ntc_temp: np.ndarray
    vsys: np.ndarray
    die_temp: np.ndarray
    iba_meas_status: np.ndarray
    buck_status: np.ndarray
    mode: np.ndarray

def load_measured_data(data_file_path: Path) -> BatteryAnalysisData:

    if not data_file_path.is_file():
        return None

    profile_data = pd.read_csv(data_file_path)

    time_vector           = profile_data["time"].to_numpy()
    vbat_vector           = profile_data["vbat"].to_numpy()
    ibat_vector           = profile_data["ibat"].to_numpy()
    ntc_temp_vector       = profile_data["ntc_temp"].to_numpy()
    vsys_vector           = profile_data["vsys"].to_numpy()
    die_temp              = profile_data["die_temp"].to_numpy()
    iba_meas_status_vecor = profile_data["iba_meas_status"].to_numpy()
    buck_status_vector    = profile_data["buck_status"].to_numpy()
    mode_vector           = profile_data["mode"].to_numpy()

    return BatteryAnalysisData(
        time=time_vector,
        vbat=vbat_vector,
        ibat=ibat_vector,
        ntc_temp=ntc_temp_vector,
        vsys=vsys_vector,
        die_temp=die_temp,
        iba_meas_status=iba_meas_status_vecor,
        buck_status=buck_status_vector,
        mode=mode_vector
    )

def export_profile_data(data: BatteryAnalysisData, output_file_path: Path):
    """
    Export the battery analysis data to a CSV file.

    Args:
        data: BatteryAnalysisData object containing the data to export
        output_file_path: Path where to save the CSV file
    """
    # Create a dictionary with all data fields
    data_dict = {
        "time": data.time,
        "vbat": data.vbat,
        "ibat": data.ibat,
        "ntc_temp": data.ntc_temp,
        "vsys": data.vsys,
        "die_temp": data.die_temp,
        "iba_meas_status": data.iba_meas_status,
        "buck_status": data.buck_status,
        "mode": data.mode
    }

    # Convert to DataFrame
    df = pd.DataFrame(data_dict)

    print(f"Export profile data to {output_file_path}")

    # Export to CSV
    df.to_csv(output_file_path, index=False)


def cut_profile_data(data, interval):

    data_cut = BatteryAnalysisData(
        time=data.time[interval],
        vbat=data.vbat[interval],
        ibat=data.ibat[interval],
        ntc_temp=data.ntc_temp[interval],
        vsys=data.vsys[interval],
        die_temp=data.die_temp[interval],
        iba_meas_status=data.iba_meas_status[interval],
        buck_status=data.buck_status[interval],
        mode=data.mode[interval]
    )

    # Offset time vector to start at 0
    data_cut.time = (data_cut.time - data_cut.time[0])
    return data_cut



def cut_discharge_profile_data(data: BatteryAnalysisData) -> BatteryAnalysisData:
    discharge_indices = np.where(data.mode == "DISCHARGING")

    discharge_data = BatteryAnalysisData(
        time=data.time[discharge_indices],
        vbat=data.vbat[discharge_indices],
        ibat=data.ibat[discharge_indices],
        ntc_temp=data.ntc_temp[discharge_indices],
        vsys=data.vsys[discharge_indices],
        die_temp=data.die_temp[discharge_indices],
        iba_meas_status=data.iba_meas_status[discharge_indices],
        buck_status=data.buck_status[discharge_indices],
        mode=data.mode[discharge_indices]
    )

    # Offset time vector to start at 0
    discharge_data.time = (discharge_data.time - discharge_data.time[0])

    return discharge_data

def get_mean_temp(temp):
    mean     = sum(temp) / len(temp)
    variance = (1/mean)*np.sum((temp-mean)**2)
    return mean, variance

def _split_continous_intervals(data, indices):
    intervals = []

    start_idx = 0
    for i in range(0, len(indices[0])):

        if(i == 0):
            continue

        if(indices[0][i] - indices[0][i-1] > 1):
            intervals.append(indices[0][start_idx:i-1])
            start_idx = i

        if(i == len(indices[0])-1):
            intervals.append(indices[0][start_idx:i])

    return intervals

def split_profile_phases(data: BatteryAnalysisData):

    discharge_indices = np.where(data.mode == "DISCHARGING")
    charge_indices = np.where(np.logical_or(data.mode == "CHARGING", data.mode == "IDLE"))

    # split into continous intervals
    charge_intervals_list = _split_continous_intervals(data, charge_indices)
    discharge_intervals_list = _split_continous_intervals(data, discharge_indices)

    charge_profiles = []
    discharge_profiles = []

    for interval in charge_intervals_list:
        charge_profiles.append(cut_profile_data(data, interval))

    for interval in discharge_intervals_list:
        discharge_profiles.append(cut_profile_data(data, interval))

    return charge_profiles, discharge_profiles


def get_complementary_color(color: str) -> str:
    # Convert the color to RGB
    rgb = mcolors.to_rgb(color)
    # Calculate the complementary color
    comp_rgb = (1.0 - rgb[0], 1.0 - rgb[1], 1.0 - rgb[2])
    # Convert back to a hex string
    return mcolors.to_hex(comp_rgb)

def print_profile(ax, ax_secondary, data: BatteryAnalysisData, label: str, color: str):

    ln = ax.plot(data.time, data.vbat, label=f"{label} vbat", color=color)
    ln_s = ax_secondary.plot(data.time, data.ibat, label=f"{label} ibat", color=get_complementary_color(color))


