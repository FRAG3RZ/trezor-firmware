

import argparse
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from pathlib import Path
import argcomplete

parser = argparse.ArgumentParser(
                    prog='analyze_charging_profile')

parser.add_argument('-f', '--input_file', help="Charging profile data file", required=True)

argcomplete.autocomplete(parser)

def low_pass_ma_filter(data_vector, filter_length):

    filtered_vector = copy.copy(data_vector)
    #
    for i, sample in enumerate(data_vector):
        if(i < filter_length):
            filtered_vector[i] = sum(data_vector[:i]/(filter_length))
        else:
            filtered_vector[i] = sum(data_vector[i-filter_length:i]/(filter_length))

    return filtered_vector

def main(args):

    ch_prof_file = Path(args.input_file)

    profile_data = pd.read_csv(ch_prof_file)

    time_vector           = profile_data["time"].to_numpy()
    vbat_vector           = profile_data["vbat"].to_numpy()
    ibat_vector           = profile_data["ibat"].to_numpy()
    ntc_temp_vector       = profile_data["ntc_temp"].to_numpy()
    vsys_vector           = profile_data["vsys"].to_numpy()
    iba_meas_status_vecor = profile_data["iba_meas_status"].to_numpy()
    buck_status_vector    = profile_data["buck_status"].to_numpy()
    mode_vector           = profile_data["mode"].to_numpy()

    # low pass filter
    vbat_filtered = low_pass_ma_filter(vbat_vector, 30)
    ibat_filtered = low_pass_ma_filter(ibat_vector, 30)

    # Offset time vector, redo to minutes
    time_vector = ((time_vector - time_vector[0]) / 60000)

    fix, ax = plt.subplots()

    ln11 = ax.plot(time_vector, vbat_vector, label="vbat", color="blue")
    ln12 = ax.plot(time_vector, vbat_filtered, label="vbat_filtered", color="red")
    ax.set_xlabel("Charging time [minutes]")
    ax.set_ylabel("Vbat [V]")
    ax2 = ax.twinx()
    ln21 = ax2.plot(time_vector, -ibat_vector, label="ibat", color="orange")
    ln22 = ax2.plot(time_vector, -ibat_filtered, label="ibat_filtered", color="red")
    ax2.set_ylabel("Ibat [mA]")
    lns = ln11 + ln12 + ln21 + ln22
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    ylim = ax.get_ylim()

    # Plot PMIC modes
    ax.fill_between(time_vector, y1=ylim[1],y2=ylim[0], where= mode_vector=="CHARGING", facecolor="green",alpha=.3)
    ax.fill_between(time_vector, y1=ylim[1],y2=ylim[0], where= mode_vector=="DISCHARGING", facecolor="orange",alpha=.3)
    ax.fill_between(time_vector, y1=ylim[1],y2=ylim[0], where= mode_vector=="IDLE", facecolor="blue",alpha=.3)

    plt.show()


if __name__ == "__main__":

    args = parser.parse_args()

    main(args)



