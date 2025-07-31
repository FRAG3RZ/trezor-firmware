
import matplotlib.pyplot as plt
from fuel_gauge.profile_data_utils import *
from fuel_gauge.battery_profiling import *
from matplotlib import style

# Load linear discharge profiles from dataset

dataset_path = Path("dataset/202502_phd/device_1/linear")
discharge_curves = dataset_path.rglob("discharge.linear.*.csv")

fig, ax = plt.subplots()
fig.set_size_inches(6,4)
fig.set_dpi(300)

plt.rcParams['pdf.fonttype'] = 42  # Ensures text is editable in PDF
plt.rcParams['font.size'] = 10     # Good base font size for readability

for i, crv_path in enumerate(sorted(discharge_curves)):

    data = load_measured_data(crv_path)

    # Calcultate accumulated charge
    total_charge_mAh = coulomb_counter(data.time, data.ibat)

    time_min = time_to_minutes(data.time)
    ax.plot(time_min, data.vbat, label=f"{crv_path.stem}", linewidth=0.5)
    ax.vlines(x=time_min[-1], ymin=min(data.vbat), ymax=3.6, color='gray', linestyle='--', linewidth=0.5)
    ax.text(x=time_min[-1]+2, y=3.3+(0.02*(i%2)), s=f"{time_min[-1]:.2f} min / {total_charge_mAh:.2f} mAh", fontsize=6, color="gray")
    print(f"{crv_path.stem}\t total_capacity: {total_charge_mAh:.2f}mAh \t discharge_time: {time_min[-1]:.2f} minutes")

left, right = ax.get_xlim()
ax.hlines(y=3.0, xmin=0, xmax=right*1.1, color='r', linestyle='--', linewidth=0.5)
ax.text(x=right-5, y=3.01, s=f"V_term", fontsize=6, color="red")
ax.set_xlabel("time [minutes]")
ax.set_ylabel("terminal_voltage [V]")
ax.set_title("Discharge curves, constant load scenario (constant backlight: 100 ~ 130mA)")
ax.legend(fontsize=6, ncol=2)  # Smaller font, 2 columns

plt.show()
