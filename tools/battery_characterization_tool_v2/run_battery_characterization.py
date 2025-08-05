

"""
This script will process the data from given battery characterization
and perform battery model identification, SOC curve extraction and
create .h lookup tables for embedded software implementation.
"""

import os
import matplotlib.pyplot as plt
from fuel_gauge.fuel_gauge import *
from fuel_gauge.profile_data_utils import *
from archive.battery_profiling import *
from fuel_gauge.battery_model import *
from fuel_gauge.coulomb_counter_estimator import *
from fuel_gauge.ekf_estimator import *
from fuel_gauge.dummy_estimator import *
from fuel_gauge.simulator import *
from generate_battery_libraries_v2 import *

DATASET="DV_phase_stable"

"""
Those values are the characteristic temperatures for which the battery profiles
were measured, remeber that this temperature is not the temeprature of the battery
but the temperature of the chamber where the battery was tested. During the characterization
NTC battery data should be used to correctly calibrate the lookup table.

"""
characterized_temperatures_deg = [15,20,25,30]
soc_curve_points_num = 100
soc_curve_max_chg_voltage = 3.6
soc_curve_max_dchg_voltage = 3.0

dataset_path = Path("dataset") / DATASET

def main():

    # Look for device device folders
    devices = [d for d in dataset_path.iterdir() if d.is_dir()]

    # Identification output
    #  Evaluated temp | Real NFC temp (average) | Estimated Rint | Extracted SOC curve |
    #  SOC curve characteristics {max_chg_voltage, max_dischg_voltage, num_of_points} |
    #  Linear discharge calculated capacity and yield,
    soc_curves = {}
    ocv_curves = {}
    r_int_points = []
    r_int_points_charge = []

    # Estimate Rint for every temperature
    for temp in characterized_temperatures_deg:

        profiles = []

        for d in devices:

            if(temp < 0):
                file_id = f"min_{abs(temp)}"
            else:
                file_id = temp

            discharge_profile = load_measured_data(d / "switching" / f"discharge.switching.{file_id}_deg.csv")

            if not discharge_profile:
                print(f"File {d / 'switching' / f'discharge.switching.{file_id}_deg.csv'} does not exist, skipping")
                continue

            R_int_estim = estimate_R_int(discharge_profile.time,
                                         discharge_profile.ibat,
                                         discharge_profile.vbat,
                                         discharge_profile.ntc_temp,
                                         debug=False)

            charge_profile = load_measured_data(d / "switching" / f"charge.switching.{file_id}_deg.csv")
            if not charge_profile:
                print(f"File {d / 'switching' / f'charge.switching.{file_id}_deg.csv'} does not exist, skipping")
                continue


            R_int_estim_charge = estimate_R_int(charge_profile.time,
                                         charge_profile.ibat,
                                         charge_profile.vbat,
                                         charge_profile.ntc_temp,
                                         debug=False)

            # Load linear discharge profile
            discharge_profile = load_measured_data(d / "linear" / f"discharge.linear.{file_id}_deg.csv")
            if not discharge_profile:
                print(f"File {d / 'linear' / f'discharge.linear.{file_id}_deg.csv'} does not exist, skipping")
                continue

            SoC_curve, total_capacity, effective_capacity = extract_SoC_curve(file_id,
                                                                        discharge_profile,
                                                                        R_int_estim,
                                                                        max_chg_voltage=soc_curve_max_chg_voltage,
                                                                        max_dischg_voltage=soc_curve_max_dchg_voltage,
                                                                        num_of_points=soc_curve_points_num,
                                                                        debug=True)

            charge_profile = load_measured_data(d / "linear" / f"charge.linear.{file_id}_deg.csv")
            if not charge_profile:
                print(f"File {d / 'linear' / f'charge.linear.{file_id}_deg.csv'} does not exist, skipping")
                continue


            # Extract SOC curve, but use the estimated iinternal resistance from discharge tests,
            # Charging swithcing tests seems to be irrelevant
            SoC_curve_charge, total_capacity_charge, effective_capacity_charge = extract_SoC_curve_charging(file_id,
                                                                    charge_profile,
                                                                    R_int_estim,
                                                                    max_chg_voltage=3.9,
                                                                    max_dischg_voltage=3.0,
                                                                    num_of_points=soc_curve_points_num,
                                                                    debug=True)


            mean_temp, _ = get_mean_temp(discharge_profile.ntc_temp)
            r_int_points.append([mean_temp, R_int_estim])
            r_int_points_charge.append([mean_temp, R_int_estim_charge])

            # Capture single profile into dict
            entry = {}
            entry["data"] = discharge_profile
            entry["ambient_temp"] = temp
            entry["ntc_temp"], _ = get_mean_temp(discharge_profile.ntc_temp)
            entry["R_int"] = R_int_estim
            entry["R_int_charge"] = R_int_estim_charge
            entry["max_chg_voltage"] = soc_curve_max_chg_voltage
            entry["max_disch_voltage"] = soc_curve_max_dchg_voltage
            entry["SoC_curve"] = SoC_curve
            entry["SoC_curve_charge"] = SoC_curve_charge
            entry['total_capacity'] = effective_capacity
            entry['total_capacity_charge'] = effective_capacity_charge
            entry['capacity_yield'] = total_capacity - effective_capacity
            profiles.append(entry)

        SoC_mean = profiles[0]["SoC_curve"][1]
        ntc_temp_mean = profiles[0]["ntc_temp"]

        soc_profiles = np.array([profiles[0]["SoC_curve"][0], profiles[0]["SoC_curve"][1]])

        for i in range(1, len(profiles)):

            prf = np.array([profiles[i]["SoC_curve"][0], profiles[i]["SoC_curve"][1]])

            soc_profiles = np.hstack((soc_profiles, prf))

            SoC_mean = np.vstack((SoC_mean, profiles[i]["SoC_curve"][1]))
            print(f"NTC temp distanve from mean {ntc_temp_mean-profiles[i]["ntc_temp"]}")
            ntc_temp_mean += profiles[i]["ntc_temp"]

        curve_params, curve_params_complete = fit_soc_curve(soc_profiles)


        soc_profiles_charge = np.array([profiles[0]["SoC_curve_charge"][0], profiles[0]["SoC_curve_charge"][1]])

        for i in range(1, len(profiles)):

            prf = np.array([profiles[i]["SoC_curve_charge"][0], profiles[i]["SoC_curve_charge"][1]])
            soc_profiles_charge = np.hstack((soc_profiles_charge, prf))

        curve_params_charge, curve_params_charge_complete = fit_soc_curve(soc_profiles_charge)

        if(len(profiles) > 1):
            SoC_mean = np.array([profiles[0]["SoC_curve"][0], np.mean(SoC_mean, axis=0)])
        else:
            SoC_mean = np.array([profiles[0]["SoC_curve"][0], SoC_mean[0]])

        ntc_temp_mean = ntc_temp_mean / len(profiles)

        # Average total capacity
        total_capacity = 0
        total_capacity_charge = 0
        capacity_yield = 0
        for p in profiles:
            print(f"Total capacity distance from mean {total_capacity - p['total_capacity']}")
            total_capacity += p['total_capacity']
            capacity_yield += p['capacity_yield']
            total_capacity_charge += p['total_capacity_charge']

        total_capacity = total_capacity / len(profiles)
        capacity_yield = capacity_yield / len(profiles)
        total_capacity_charge = total_capacity_charge / len(profiles)

        R_int = 0
        for p in profiles:
            print(f"Total R int distance from mean {R_int-p['R_int']}")
            R_int += p['R_int']
        R_int_estim = R_int / len(profiles)

        ocv_curves[round(ntc_temp_mean,2)] = {}
        ocv_curves[round(ntc_temp_mean,2)]['ocv_discharge'] = curve_params_complete
        ocv_curves[round(ntc_temp_mean,2)]['ocv_charge'] = curve_params_charge_complete
        ocv_curves[round(ntc_temp_mean,2)]['total_capacity'] = total_capacity
        ocv_curves[round(ntc_temp_mean,2)]['total_capacity_charge'] = total_capacity_charge

        soc_curves[round(ntc_temp_mean,2)] = {}
        soc_curves[round(ntc_temp_mean,2)]['curve'] = SoC_mean
        soc_curves[round(ntc_temp_mean,2)]['soc_params'] = curve_params
        soc_curves[round(ntc_temp_mean,2)]['soc_params_charge'] = curve_params_charge
        soc_curves[round(ntc_temp_mean,2)]['r_int'] = R_int_estim
        soc_curves[round(ntc_temp_mean,2)]['total_capacity'] = total_capacity
        soc_curves[round(ntc_temp_mean,2)]['capacity_yield'] = capacity_yield

    print(ocv_curves)

    # Fit R_int estimate
    r_int_vector = np.transpose(np.array(r_int_points))
    r_int_curve_params = fit_R_int_curve(r_int_vector[1], r_int_vector[0])

    r_int_vector_charge = np.transpose(np.array(r_int_points_charge))
    r_int_curve_params_charge = fit_R_int_curve(r_int_vector_charge[1], r_int_vector_charge[0])

    # Plot rint curves
    fig, ax = plt.subplots()
    fig.set_size_inches(6,4)

    fig.set_dpi(300)
    plt.rcParams['pdf.fonttype'] = 42  # Ensures text is editable in PDF
    plt.rcParams['font.size'] = 6      # Good base font size for readability
    ax.set_title(f"Rint curve params {r_int_curve_params}", wrap=True)
    ax.set_xlabel("Temperature [$^\circ$C]", fontsize=6)
    ax.set_ylabel("Rin [$\Omega$]", fontsize=6)
    ax.plot(r_int_vector[0], r_int_vector[1], marker='+', label="Rint estimation")
    ax.plot(r_int_vector_charge[0], r_int_vector_charge[1], marker='+', label="Rint estimation charge")
    ax.plot(r_int_vector[0], rational_fit(r_int_vector[0], *r_int_curve_params), label="Rint curve fit")
    ax.plot(r_int_vector_charge[0], rational_fit(r_int_vector_charge[0], *r_int_curve_params_charge), label="Rint curve fit charge")
    ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.set_xlim(xmin=0,xmax=50)

    battery_model_data = {}
    battery_model_data['r_int'] = r_int_curve_params
    battery_model_data['ocv_curves'] = ocv_curves

    # Tune fuel gauge and run simulation for all cases.
    print(f"Rint curve params {r_int_curve_params}")

    # Put together the battery model
    battery_model = BatteryModel(battery_model_data, "test hash")

    # export_battery_model_lookup(battery_model_data, output_dir="generated_output")

    generate_battery_libraries(battery_model_data, output_dir="exported_libraries_updated", battery_name="JYHPFL333838")
    """
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    Identification plot export.
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """

    export_dir_path = Path("images") / DATASET
    if not os.path.exists(export_dir_path):
        os.makedirs(export_dir_path)


    lin_space = np.linspace(0,1,100)

    fig, ax = plt.subplots()
    fig.set_size_inches(6,4)
    fig.set_dpi(300)

    plt.rcParams['pdf.fonttype'] = 42  # Ensures text is editable in PDF
    plt.rcParams['font.size'] = 6      # Good base font size for readability

    for p in sorted(list(soc_curves.keys())):
        ax.plot(soc_curves[p]["curve"][0],soc_curves[p]["curve"][1], label=f"{p} $^\circ$C", linewidth=0.5, alpha=0.5)
        ax.plot(lin_space, rational_linear_rational(lin_space, *soc_curves[p]["soc_params"]), label=f"{p}_discharge $^\circ$C", linewidth=1)
        ax.plot(lin_space, rational_linear_rational(lin_space, *soc_curves[p]["soc_params_charge"]), label=f"{p}_charge $^\circ$C", linewidth=1)

        ax.set_title(f"SOC/VOC curves (scaled from 3.0 - 3.6 terminal voltage, sampled with 100 points)", wrap=True)

        ax.legend()

        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.set_xlim(xmin=0,xmax=1)
        ax.set_xlabel("SoC", fontsize=6)
        ax.set_ylabel("Voc [V]", fontsize=6)

    fig.savefig(f"images/{DATASET}/{DATASET}_soc_voc_curves.pdf")
    fig.savefig(f"images/{DATASET}/{DATASET}_soc_voc_curves.png")

    fig, ax = plt.subplots()
    fig.set_size_inches(6,4)
    fig.set_dpi(300)

    plt.rcParams['pdf.fonttype'] = 42  # Ensures text is editable in PDF
    plt.rcParams['font.size'] = 6      # Good base font size for readability

    ax.set_title(f"Internal resistance evolution", wrap=True)

    r_in = []
    temp = []

    for p in sorted(list(soc_curves.keys())):
        r_in.append(soc_curves[p]["r_int"])
        temp.append(p)

    r_in = np.array(r_in)
    temp = np.array(temp)

    ax.plot(temp, r_in, marker='+', label=f"{p} $^\circ$C")
    ax.plot(temp, rational_fit(temp, *r_int_curve_params), label=f"{p} $^\circ$C", linewidth=1)

    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.set_xlabel("Temperature [$^\circ$C]", fontsize=6)
    ax.set_ylabel("Rin [$\Omega$]", fontsize=6)

    fig.savefig(f"images/{DATASET}/{DATASET}_R_in_evolution.pdf")
    fig.savefig(f"images/{DATASET}/{DATASET}_R_in_evolution.png")

    """
    Q: Defines the covariance of the process noise, AKA, how much you trust the model.

    R: Defines the covariance of the observation noise, AKA, how much you trust your sensors (higher the number, less the trust)

    P_init defines confidence to your initial guess, smaller the number higher the confidence
    should be set small enough to not make an unncecessary transition effect in the beggining of the simulation (fuel gauge is uncertain and go
    crazy to find the correct SoC), but not too small to make the fuel gauge "too sturdy" and react dynamically on incorrect measurement
    """
    ukf_model = fuel_gauge(soc_curves, model_type="ukf", Q=0.001, R=1000, n=1, alpha=1.0, beta=1.0, kappa=0, P_init=0.001)
    ukf_dynamic_model = fuel_gauge(soc_curves, model_type="ukf", Q=0.001, R=100, n=1, alpha=2.0, beta=2.0, kappa=0, P_init=0.001)
    dir_model = fuel_gauge(soc_curves, model_type="direct")
    ekf_model = fuel_gauge(soc_curves, model_type="ekf", Q=0.05, R=20, P_init=100)

    ekf_model_adaptive_ref = fuel_gauge(soc_curves, model_type="ekf_adaptive", Q=0.001, R=3000, Q_agressive=0.001, R_agressive=3000, P_init=0.1)
    ekf_model_adaptive = fuel_gauge(soc_curves, model_type="ekf_adaptive", Q=0.001, R=3000, Q_agressive=0.001,  R_agressive=3000, P_init=0.1)

    # Simulation data
    simulation_data = dataset_path.rglob("discharge.*.csv")

    discharge_profile = load_measured_data(Path("dataset") / "202502_phd" / "device_1" / "switching" / f"discharge.switching.25_deg.csv")
    # sim_time_ekf, soc_ekf, sim_start_ekf, sim_end_ekf = ekf_model.run_simulation(discharge_profile, sp=0, override_init_soc=None)

    fig, ax = plt.subplots()

    # ax.plot(discharge_profile.time[sim_start_ekf:sim_end_ekf], soc_ekf[sim_start_ekf:sim_end_ekf], label="EKF")
    # plt.show()

    coulomb_counter_est = CoulombCounterEstimator(battery_model=battery_model)
    dummy_est = DummyEstimator(battery_model=battery_model)
    ekf_est = EkfEstimator(battery_model=battery_model, R=2000, Q=0.001, Q_agressive=0.001, R_agressive=1000, P_init=0.1)
    ekf_est2 = EkfEstimator(battery_model=battery_model, R=2000, Q=0.001, Q_agressive=0.001, R_agressive=1000, P_init=0.1)

    error = []

    init_soc = 0.75

    for sim in simulation_data:

        print(sim)
        discharge_profile = load_measured_data(sim)

        filtered_profile = copy.copy(discharge_profile)
        filter_length = 80
        filtered_profile.vbat = low_pass_ma_filter(discharge_profile.vbat, filter_length)
        filtered_profile.ibat = low_pass_ma_filter(discharge_profile.ibat, filter_length)

        sp = int(len(discharge_profile.time) * 0.25)

        cc_soc, _, cc_sim_time, cc_sim_start, cc_sim_end = run_battery_simulation(discharge_profile,
                                                                                  coulomb_counter_est)

        dummy_soc, _, dummy_sim_time, dummy_sim_start, dummy_sim_end = run_battery_simulation(discharge_profile,
                                                                                  dummy_est)

        ekf_soc, ekf_cov, ekf_sim_time, ekf_sim_start, ekf_sim_end = run_battery_simulation(discharge_profile,
                                                                                            ekf_est)

        ekf_soc2, ekf_cov2, ekf_sim_time2, ekf_sim_start2, ekf_sim_end2 = run_battery_simulation(discharge_profile,
                                                                                            ekf_est2, sim_start_idx=sp)

        # sim_time_ukf, soc_ukf, sim_start_ukf, sim_end_ukf = ukf_model.run_simulation(disfiltered_profilecharge_profile, sp=sp, override_init_soc=None)
        # sim_time_ekf, soc_ekf, sim_start_ekf, sim_end_ekf = ekf_model.run_simulation(discharge_profile, sp=sp, override_init_soc=init_soc)
        sim_time_dir, soc_dir, sim_start_dir, sim_end_dir = dir_model.run_simulation(discharge_profile, sp=sp, override_init_soc=init_soc)
        sim_time_fir, soc_fir, sim_start_fir, sim_end_fir = dir_model.run_simulation(filtered_profile, sp=0, override_init_soc=init_soc)


        sim_time_ekf_a, soc_ekf_a, sim_start_ekf_a, sim_end_ekf_a = ekf_model_adaptive.run_simulation(filtered_profile, sp=sp, override_init_soc=None, init_filter=True)
        sim_time_ekf_ar, soc_ekf_ar, sim_start_ekf_ar, sim_end_ekf_ar = ekf_model_adaptive_ref.run_simulation(discharge_profile, sp=0, override_init_soc=None, init_filter=True)
        #sim_time_ukf_d, soc_ukf_d, sim_start_ukf_d, sim_end_ukf_d = ukf_dynamic_model.run_simulation(discharge_profile, sp=sp, override_init_soc=init_soc)

        # total_charge_ukf = coulomb_counter(discharge_profile.time[sim_start_ukf:sim_end_ukf] , discharge_profile.ibat[sim_start_ukf:sim_end_ukf])
        # total_charge_ekf = coulomb_counter(discharge_profile.time[sim_start_ekf:sim_end_ekf] , discharge_profile.ibat[sim_start_ekf:sim_end_ekf])
        total_charge_dir = coulomb_counter(discharge_profile.time[sim_start_dir:sim_end_dir] , discharge_profile.ibat[sim_start_dir:sim_end_dir])

        voltage_threshold = 3.0
        for i in reversed(range(0, len(discharge_profile.vbat))):
            if discharge_profile.vbat[i] > voltage_threshold:
                UVLO = i
                break

        # Define simulation error as time distance between vbat voltage drop under 3V and SoC drop under 0%
        # error.append((discharge_profile.time[UVLO]/1000) - (discharge_profile.time[sim_end_ukf]/1000))

        fig, ax = plt.subplots(4,1)
        fig.set_size_inches(10, 10)

        #fig.set_dpi(300)

        #plt.rcParams['pdf.fonttype'] = 42  # Ensures text is editable in PDF
        #plt.rcParams['font.size'] = 6      # Good base font size for readability

        fig.suptitle(f"{sim.name}")
        ax[0].plot(time_to_minutes(discharge_profile.time), discharge_profile.vbat, label="Vbat")
        ax[0].plot(time_to_minutes(discharge_profile.time), discharge_profile.vbat, label="Vbat_filtered")
        # ax[0].plot(discharge_profile.time[sim_start_dir:sim_end_dir], term_voc[sim_start_dir:sim_end_dir], label="termination_voltage")

        ceil, floor  = ax[0].get_ylim()
        #ax[0].vlines(discharge_profile.time[sim_end_ukf], ceil, floor, color="black", label="25%", alpha=0.3)
        ax[0].vlines(time_to_minutes(discharge_profile.time[UVLO], discharge_profile.time[0]), ceil, floor, color="red", label="UVLO")
        ax[0].set_title("Battery voltage")

        ax[0].legend()
        ax[0].set_xlim((0, time_to_minutes(discharge_profile.time[-1], discharge_profile.time[0])))

        ax[1].plot(time_to_minutes(discharge_profile.time), discharge_profile.ibat)
        ax[1].plot(time_to_minutes(filtered_profile.time), filtered_profile.ibat)

        ax[1].set_title("Battert current")
        ax[1].set_xlim((0, time_to_minutes(discharge_profile.time[-1], discharge_profile.time[0])))
        ax[2].plot(time_to_minutes(discharge_profile.time[sim_start_dir:sim_end_dir],discharge_profile.time[0]), soc_dir[sim_start_dir:sim_end_dir], label=f"Fuel_gauge Direct {total_charge_dir:.2f}mAh", alpha=0.3)
        ax[2].plot(time_to_minutes(filtered_profile.time[sim_start_fir:sim_end_fir],discharge_profile.time[0]), soc_fir[sim_start_fir:sim_end_fir], label=f"Fuel_gauge Direct filtered {total_charge_dir:.2f}mAh", alpha=0.4)

        ax[2].plot(time_to_minutes(discharge_profile.time[cc_sim_start:cc_sim_end], discharge_profile.time[0]), cc_soc[cc_sim_start:cc_sim_end], label=f"Coulomb counter estimator")
        ax[2].plot(time_to_minutes(discharge_profile.time[dummy_sim_start:dummy_sim_end], discharge_profile.time[0]), dummy_soc[dummy_sim_start:dummy_sim_end], label=f"Dummy estimator")
        ax[2].plot(time_to_minutes(discharge_profile.time[ekf_sim_start:ekf_sim_end], discharge_profile.time[0]), ekf_soc[ekf_sim_start:ekf_sim_end],marker='.', linestyle='-',label=f"EKF estimator")
        ax[2].plot(time_to_minutes(discharge_profile.time[ekf_sim_start2:ekf_sim_end2], discharge_profile.time[0]), ekf_soc2[ekf_sim_start2:ekf_sim_end2], label=f"EKF2 estimator")


        ax[2].set_title("SoC")
        ax[2].legend()
        ax[2].set_xlim((0, time_to_minutes(discharge_profile.time[-1], discharge_profile.time[0])))
        left, right = ax[2].get_xlim()
        ax[2].hlines(0, left, right, color="black", label="max charge voltage", alpha=0.3)
        ax[2].hlines(0.25, left, right, color="black", label="max charge voltage", alpha=0.3)
        ax[2].hlines(0.5, left, right, color="black", label="max charge voltage", alpha=0.3)
        ax[2].hlines(0.75, left, right, color="black", label="max charge voltage", alpha=0.3)
        ax[2].hlines(1, left, right, color="black", label="max charge voltage", alpha=0.3)
        ax[2].vlines(time_to_minutes(discharge_profile.time[int(UVLO*0.25)], discharge_profile.time[0]), 0, 1, color="black", label="25%", alpha=0.3)
        ax[2].vlines(time_to_minutes(discharge_profile.time[int(UVLO*0.50)], discharge_profile.time[0]), 0, 1, color="black", label="50%", alpha=0.3)
        ax[2].vlines(time_to_minutes(discharge_profile.time[int(UVLO*0.75)], discharge_profile.time[0]), 0, 1, color="black", label="75%", alpha=0.3)
        ax[2].vlines(time_to_minutes(discharge_profile.time[UVLO], discharge_profile.time[0]), 0, 1, color="red", label="UVLO")

        ax[3].plot(time_to_minutes(discharge_profile.time), discharge_profile.ntc_temp)
        ax[3].set_title("NTC temperature")

        plt.savefig(f"regression_results/{sim.parent.parent.name}.{sim.name}.pdf")

    simulation_data = dataset_path.rglob("charge.*.csv")

    ekf_model_adaptive_ref = fuel_gauge(soc_curves, model_type="ekf_adaptive", Q=100, R=1, Q_agressive=1, R_agressive=1, P_init=10000)
    ekf_model_adaptive = fuel_gauge(soc_curves, model_type="ekf_adaptive", Q=100, R=1, Q_agressive=1,  R_agressive=1, P_init=10000)

    plt.show()

    sp = 0
    init_soc = 0.25


    for sim in simulation_data:

        print(sim)
        discharge_profile = load_measured_data(sim)

        filtered_profile = copy.copy(discharge_profile)
        filter_length = 80
        filtered_profile.vbat = low_pass_ma_filter(discharge_profile.vbat, filter_length)
        filtered_profile.ibat = low_pass_ma_filter(discharge_profile.ibat, filter_length)

        sp = int(len(discharge_profile.time) * 0.25)

        cc_soc, _, cc_sim_time, cc_sim_start, cc_sim_end = run_battery_simulation(discharge_profile,
                                                                                  coulomb_counter_est)

        dummy_soc, _, dummy_sim_time, dummy_sim_start, dummy_sim_end = run_battery_simulation(discharge_profile,
                                                                                  dummy_est)

        ekf_soc, ekf_cov, ekf_sim_time, ekf_sim_start, ekf_sim_end = run_battery_simulation(discharge_profile,
                                                                                            ekf_est)

        ekf_soc2, ekf_cov2, ekf_sim_time2, ekf_sim_start2, ekf_sim_end2 = run_battery_simulation(discharge_profile,
                                                                                            ekf_est2, sim_start_idx=sp)


        sim_time_dir, soc_dir, sim_start_dir, sim_end_dir = dir_model.run_simulation(discharge_profile, sp=sp, override_init_soc=init_soc)
        sim_time_fir, soc_fir, sim_start_fir, sim_end_fir = dir_model.run_simulation(filtered_profile, sp=0, override_init_soc=init_soc)

        total_charge_dir = coulomb_counter(discharge_profile.time[sim_start_dir:sim_end_dir] , discharge_profile.ibat[sim_start_dir:sim_end_dir])

        voltage_threshold = 3.0
        for i in reversed(range(0, len(discharge_profile.vbat))):
            if discharge_profile.vbat[i] > voltage_threshold:
                UVLO = i
                break

        # Define simulation error as time distance between vbat voltage drop under 3V and SoC drop under 0%
        # error.append((discharge_profile.time[UVLO]/1000) - (discharge_profile.time[sim_end_ukf]/1000))

        fig, ax = plt.subplots(4,1)
        fig.set_size_inches(10, 10)

        #fig.set_dpi(300)

        #plt.rcParams['pdf.fonttype'] = 42  # Ensures text is editable in PDF
        #plt.rcParams['font.size'] = 6      # Good base font size for readability

        fig.suptitle(f"{sim.name}")
        ax[0].plot(time_to_minutes(discharge_profile.time), discharge_profile.vbat, label="Vbat")
        ax[0].plot(time_to_minutes(discharge_profile.time), discharge_profile.vbat, label="Vbat_filtered")
        # ax[0].plot(discharge_profile.time[sim_start_dir:sim_end_dir], term_voc[sim_start_dir:sim_end_dir], label="termination_voltage")

        ceil, floor  = ax[0].get_ylim()
        #ax[0].vlines(discharge_profile.time[sim_end_ukf], ceil, floor, color="black", label="25%", alpha=0.3)
        ax[0].vlines(time_to_minutes(discharge_profile.time[UVLO], discharge_profile.time[0]), ceil, floor, color="red", label="UVLO")
        ax[0].set_title("Battery voltage")

        ax[0].legend()
        ax[0].set_xlim((0, time_to_minutes(discharge_profile.time[-1], discharge_profile.time[0])))

        ax[1].plot(time_to_minutes(discharge_profile.time), discharge_profile.ibat)
        ax[1].plot(time_to_minutes(filtered_profile.time), filtered_profile.ibat)

        ax[1].set_title("Battert current")
        ax[1].set_xlim((0, time_to_minutes(discharge_profile.time[-1], discharge_profile.time[0])))

        ax[2].plot(time_to_minutes(discharge_profile.time[cc_sim_start:cc_sim_end], discharge_profile.time[0]), cc_soc[cc_sim_start:cc_sim_end], label=f"Coulomb counter estimator")
        ax[2].plot(time_to_minutes(discharge_profile.time[dummy_sim_start:dummy_sim_end], discharge_profile.time[0]), dummy_soc[dummy_sim_start:dummy_sim_end], label=f"Dummy estimator")
        ax[2].plot(time_to_minutes(discharge_profile.time[ekf_sim_start:ekf_sim_end], discharge_profile.time[0]), ekf_soc[ekf_sim_start:ekf_sim_end], label=f"EKF estimator")
        ax[2].plot(time_to_minutes(discharge_profile.time[ekf_sim_start2:ekf_sim_end2], discharge_profile.time[0]), ekf_soc2[ekf_sim_start2:ekf_sim_end2], label=f"EKF2 estimator")

        ax[2].set_title("SoC")
        ax[2].legend()
        ax[2].set_xlim((0, time_to_minutes(discharge_profile.time[-1], discharge_profile.time[0])))
        left, right = ax[2].get_xlim()
        ax[2].hlines(0, left, right, color="black", label="max charge voltage", alpha=0.3)
        ax[2].hlines(0.25, left, right, color="black", label="max charge voltage", alpha=0.3)
        ax[2].hlines(0.5, left, right, color="black", label="max charge voltage", alpha=0.3)
        ax[2].hlines(0.75, left, right, color="black", label="max charge voltage", alpha=0.3)
        ax[2].hlines(1, left, right, color="black", label="max charge voltage", alpha=0.3)
        ax[2].vlines(time_to_minutes(discharge_profile.time[int(UVLO*0.25)], discharge_profile.time[0]), 0, 1, color="black", label="25%", alpha=0.3)
        ax[2].vlines(time_to_minutes(discharge_profile.time[int(UVLO*0.50)], discharge_profile.time[0]), 0, 1, color="black", label="50%", alpha=0.3)
        ax[2].vlines(time_to_minutes(discharge_profile.time[int(UVLO*0.75)], discharge_profile.time[0]), 0, 1, color="black", label="75%", alpha=0.3)
        ax[2].vlines(time_to_minutes(discharge_profile.time[UVLO], discharge_profile.time[0]), 0, 1, color="red", label="UVLO")

        ax[3].plot(time_to_minutes(discharge_profile.time), discharge_profile.ntc_temp)
        ax[3].set_title("NTC temperature")

        plt.savefig(f"regression_results/{sim.parent.parent.name}.{sim.name}.pdf")

    fig, ax = plt.subplots()
    # ax.plot(error)
    ax.set_ylabel("seconds")
    ax.set_xlabel("simulation_sample")
    left, right = ax.get_xlim()
    # avg_error = sum(error)/len(error)
    # ax.hlines(avg_error, left, right, color="black", label=f"avg error {avg_error} ", alpha=0.3)
    ax.legend()
    ax.set_title("Error")

    plt.show()

if __name__ == "__main__":
    main()














