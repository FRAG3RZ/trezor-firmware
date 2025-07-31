
from fuel_gauge.profile_data_utils import *
from fuel_gauge.fuel_gauge import *
import matplotlib.pyplot as plt


profile_medium = load_measured_data("data/charge_discharge_middle_load.csv")
profile_heavy  = load_measured_data("data/charge_discharge_heavy_load.csv")
profile_switching = load_measured_data("data/charge_discharge_load_switching_test.csv")
profile_d34_4 = load_measured_data("device34/charge_discharge_load_switching_4.csv")
profile_d34_5 = load_measured_data("device34/charge_discharge_load_switching_5.csv")
profile_d34_6 = load_measured_data("device34/charge_discharge_load_switching_6.csv")

# Data from temperature chamber
d1_discharge_sw_25 = load_measured_data("battery_characterization/raw_data/device_1/discharge.switching.25_deg.csv")
d2_discharge_sw_25 = load_measured_data("battery_characterization/raw_data/device_2/discharge.switching.25_deg.csv")

profile_medium_dis = cut_discharge_profile_data(profile_medium)
profile_heavy_dis = cut_discharge_profile_data(profile_heavy)
profile_switching_dis = cut_discharge_profile_data(profile_switching)
profile_d34_4_dis = cut_discharge_profile_data(profile_d34_4)
profile_d34_5_dis = cut_discharge_profile_data(profile_d34_5)
profile_d34_6_dis = cut_discharge_profile_data(profile_d34_6)
profile_d1_discharge_sw_25 = cut_discharge_profile_data(d1_discharge_sw_25)
profile_d2_discharge_sw_25 = cut_discharge_profile_data(d2_discharge_sw_25)


"""
Select the discharge profile used for model identifiaction and the profile used for regression
"""
identification_profile = profile_d1_discharge_sw_25
regression_profile = profile_d2_discharge_sw_25


max_chg_voltage = 3.6
max_dischg_voltage = 3
num_of_soc_curve_points = 11

R_est_final = estimate_R_int(identification_profile.time, identification_profile.ibat, identification_profile.vbat, identification_profile.ntc_temp)

print(f"Estimated Rint: {R_est_final} Ohm")

# R_est_final = 0.27
SoC_curve, total_capacity = extract_SOC_curve(identification_profile, R_est_final, max_chg_voltage=max_chg_voltage, max_dischg_voltage=max_dischg_voltage, num_of_points=num_of_soc_curve_points)
print(SoC_curve)

"""
Q: Defines the covariance of the process noise, AKA, how much you trust the model.

R: Defines the covariance of the observation noise, AKA, how much you trust your sensors (higher the number, less the trust)

P_init defines confidence to your initial guess, smaller the number higher the confidence
should be set small enough to not make an unncecessary transition effect in the beggining of the simulation (fuel gauge is uncertain and go
crazy to find the correct SoC), but not too small to make the fuel gauge "too sturdy" and react dynamically on incorrect measurement
"""
fg = fuel_gauge(R_est_final, SoC_curve, total_capacity, Q=0.00, R=3000, P_init=100)

# Run simulation

# Put initial guess bit away from actual start point to see if it converges
sp = 0 # start point
fg.initial_guess(regression_profile.vbat[sp], regression_profile.ibat[sp], override_soc=0.7)

Fuel_gauge = []
P_evolution = []

for i, t in enumerate(regression_profile.time):

    if i == 0:
        continue
    SoC, P = fg.update(regression_profile.time[i]-regression_profile.time[i-1], regression_profile.vbat[i], regression_profile.ibat[i], regression_profile.ntc_temp[i])
    print(f"[{i}/{len(regression_profile.time)}] SoC: {SoC}")

    Fuel_gauge.append(SoC)
    P_evolution.append(P)

Fuel_gauge_np = np.array(Fuel_gauge)

fig, ax = plt.subplots()

ax.plot(regression_profile.time/60000, regression_profile.vbat + R_est_final*(regression_profile.ibat/1000), label="v_oc_estimated")
ax.plot(regression_profile.time/60000, regression_profile.vbat, label="vbat")
ax.plot(regression_profile.time/60000, regression_profile.ntc_temp, label="ntc_temp")
ax.plot(regression_profile.time/60000, regression_profile.die_temp, label="die_temp")
ax.set_title("Regression data")
ax.legend()

fig, ax = plt.subplots(3,1)

ax[0].plot(regression_profile.time/60000, regression_profile.vbat + R_est_final*(regression_profile.ibat/1000), label="v_oc_estimated")
ax[0].plot(regression_profile.time/60000, regression_profile.vbat, label="vbat")
ax[0].set_title("Battery voltage")
ax[0].legend()
ax[0].set_ylabel("Voltage [V]")
ax[0].set_xlabel("Time [min]")
ax[0].set_xlim(0, regression_profile.time[-1]/60000)
left, right = ax[0].get_xlim()
ax[0].hlines(max_chg_voltage, left, right, color="red", label="max charge voltage")
ax[0].hlines(max_dischg_voltage, left, right, color="red", label="max charge voltage")

up, down = ax[0].get_ylim()
ax[0].vlines((regression_profile.time[-1]/60000)/2, up, down, color="black", label="max charge voltage")

ax[1].plot(regression_profile.time[:len(Fuel_gauge)]/60000, Fuel_gauge, label="SoC")
ax[1].set_ylabel("SoC")
ax[1].set_xlabel("Time [min]")
ax[1].set_title("State of charge (SoC)")
ax[1].set_xlim(0, regression_profile.time[-1]/60000)
left, right = ax[1].get_xlim()
ax[1].hlines(0, left, right, color="black", label="max charge voltage", alpha=0.3)
ax[1].hlines(0.25, left, right, color="black", label="max charge voltage", alpha=0.3)
ax[1].hlines(0.5, left, right, color="black", label="max charge voltage", alpha=0.3)
ax[1].hlines(0.75, left, right, color="black", label="max charge voltage", alpha=0.3)
ax[1].hlines(1, left, right, color="black", label="max charge voltage", alpha=0.3)


up, down = ax[1].get_ylim()
ax[1].vlines((regression_profile.time[-1]/60000)/2, up, down, color="black", label="max charge voltage")

left, right = ax[0].get_xlim()
ax[2].plot(regression_profile.time[:len(P_evolution)]/60000, P_evolution, label="P")
ax[2].set_ylabel("P")
ax[2].set_xlabel("Time [min]")
ax[2].set_title("Covariance evolution")
ax[2].set_xlim(0, regression_profile.time[-1]/60000)
left, right = ax[0].get_xlim()
# ax.fill_betweenx(y=[0, 1], x1=used_profile.time[Fuel_gauge_np], x2=used_profile.time[len(Fuel_gauge)], color="green", alpha=0.3)

plt.show()



