import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import re
import copy
import plotly.graph_objects as go
from dataset.battery_profile import cut_charging_phase,cut_discharging_phase

DEBUG = False

estimators_to_plot = ["ekf1"]

def low_pass_ma_filter(data_vector, filter_length):

    filtered_vector = copy.copy(data_vector)

    for i, sample in enumerate(data_vector):
        if i < filter_length:
            filtered_vector[i] = sum(data_vector[: i + 1] / (i + 1))
        else:
            filtered_vector[i] = sum(
                data_vector[i - filter_length : i] / (filter_length)
            )

    # compensation = np.zeros((int(filter_length/2))) + filtered_vector[-1]

    # return np.hstack((filtered_vector[int(filter_length/2)::], compensation))
    return filtered_vector

def compute_soc_end_error(
    waveform,
    soc_from_ekf,
    threshold_v_discharge=3.1,
    ibat_idle_thresh_mA=5,
    voltage_consistency_window=5,  # increased window
    sim_name=None
):
    if not sim_name:
        return None

    sim_parts = sim_name.split(".")
    if sim_parts[2].lower() == "random_wonder":
        return None
    
    clean_data = None
    mode = sim_parts[3].lower() if len(sim_parts) > 3 else ""
    if "discharging" in mode:
        target_soc = 0
        #clean_data = cut_discharging_phase(waveform)
        threshold_v = threshold_v_discharge
    elif "charging" in mode:
        target_soc = 100
        #clean_data = cut_charging_phase(waveform)
    else:
        return None
    
    soc = soc_from_ekf
    if(clean_data is not None): vbat = clean_data.vbat 
    else: vbat = waveform.vbat
    if(clean_data is not None): ibat = clean_data.ibat 
    else: ibat = waveform.ibat

    for i in range(len(soc)):
        print(soc)
        i_mA = ibat[i]
        if abs(i_mA) >= ibat_idle_thresh_mA:
            continue  # not idle yet

        # Check voltage consistency window
        check_window_start = max(0, i - voltage_consistency_window + 1)
        v_window = vbat[check_window_start:i+1]

        if mode == "discharging" and vbat[i] > threshold_v:
            continue  
        
        # Found a valid idle point
        #print(f"Found idle point for simulation: {sim_name}")
        err = soc[i] - target_soc
        #print(f"End SoC error for {sim_name}: {err:.4f} (target: {target_soc}), (soc: {soc[i]:.4f}")
        return abs(err)

    #print(f"Warning: No valid idle point found in {sim_name} for end SoC error computation.")
    return None


def compute_soc_error(
    waveform,
    soc_from_ekf,
    start_idx, 
    end_idx, 
    battery_model,
    stability_window=2,
    current_tolerance_mA=50,
    min_current_mA=1,
    ibat_filter_length=5,
    sim_name=None
):
    abs_errors = []
    signed_errors = []
    percent_errors = []
    weights = []
    epsilon = 1e-6

    # --- Simulation name & mode validation ---
    if not sim_name:
        return None

    sim_parts = sim_name.split(".")
    
    if sim_parts[2].lower() == "random_wonder":
        return None
    
    mode = sim_parts[3].lower() if len(sim_parts) > 3 else ""
    if "discharging" in mode:
        clean_data = cut_discharging_phase(waveform)
    elif "charging" in mode:
        clean_data = cut_charging_phase(waveform)
    
    #waveform stuff
    soc = soc_from_ekf
    time = clean_data.time
    temp = clean_data.ntc_temp
    ibat = clean_data.ibat

    # Smooth the sliced ibat
    smoothed_ibat = low_pass_ma_filter(ibat[start_idx:end_idx], ibat_filter_length)
    smoothed_length = len(smoothed_ibat)

    for i in range(start_idx, start_idx + smoothed_length):
        # Convert i to index in smoothed_ibat
        smoothed_idx = i - start_idx

        if smoothed_idx < stability_window:
            continue

        # recent ibats for stability check (use smoothed array)
        recent_ibats = smoothed_ibat[smoothed_idx - stability_window:smoothed_idx]
        if len(recent_ibats) < stability_window:
            continue

        variation = max(recent_ibats) - min(recent_ibats)
        stability_penalty = 1.0 #if variation <= current_tolerance_mA else 0

        ibat_val_mA = smoothed_ibat[smoothed_idx]
        dt_ms = time[i] - time[i - 1]
        if dt_ms <= 0:
            continue

        dt = dt_ms
        soc_gradient = (soc[i] - soc[i - 1]) / dt
        ibat_val_A = ibat_val_mA / 1000.0
        temp_deg = temp[i]
        discharging = ibat_val_A > 0

        try:
            capacity_mAh = battery_model._total_capacity(temp_deg, discharging)
        except Exception:
            continue

        if capacity_mAh <= epsilon:
            continue

        capacity_As = capacity_mAh * 3.6
        expected_gradient = -ibat_val_A / capacity_As

        err = soc_gradient - expected_gradient
        abs_err = abs(err)

        abs_errors.append(abs_err)
        signed_errors.append(err)

        if abs(expected_gradient) > epsilon:
            percent_errors.append(abs_err / abs(expected_gradient))
        else:
            percent_errors.append(None)

        weight = max(abs(ibat_val_A), min_current_mA / 1000.0) * stability_penalty
        weights.append(weight)

    if not abs_errors:
        return {
            "avg_error": None,
            "rms_error": None,
            "median_error": None,
            "max_error": None,
            "avg_signed_error": None,
            "avg_percent_error": None,
            "cumulative_error": None,
            "valid_points": 0
        }

    weights = np.array(weights)
    abs_errors = np.array(abs_errors)

    avg_error = np.average(abs_errors, weights=weights)
    rms_error = (np.average(abs_errors**2, weights=weights))**0.5
    median_error = float(np.median(abs_errors))
    max_error = float(np.max(abs_errors))
    avg_signed_error = np.average(signed_errors, weights=weights)

    valid_percent_errors = [p for p in percent_errors if p is not None]
    avg_percent_error = np.average(valid_percent_errors) if valid_percent_errors else None

    cumulative_error = np.sum(abs_errors * weights)

    return {
        "avg_error": avg_error,
        "rms_error": rms_error,
        "median_error": median_error,
        "max_error": max_error,
        "avg_signed_error": avg_signed_error,
        "avg_percent_error": avg_percent_error,
        "cumulative_error": cumulative_error,
        "valid_points": len(abs_errors)
    }


def compute_linear_deviation_error(
    waveform,
    soc_from_ekf,
    start_idx, 
    end_idx,
    num_points=100,
    soc_reference_point=None,
    sim_name=None
):
    epsilon = 1e-9
    abs_errors = []

    # --- Simulation name & mode validation ---
    if not sim_name:
        return None

    sim_parts = sim_name.split(".")
    if len(sim_parts) < 3 or "linear" not in sim_parts[2].lower():
        return None

    mode = sim_parts[3].lower() if len(sim_parts) > 3 else ""
    if "discharging" in mode:
        #clean_data = cut_discharging_phase(waveform)
        soc_start = 1.0
        soc_end = 0.0
    elif "charging" in mode:
        #clean_data = cut_charging_phase(waveform)
        soc_start = 0.0
        soc_end = 1.0
    else:
        return None
    # ------------------------------------------

    soc = soc_from_ekf
    time = waveform.time
    ibat = waveform.ibat

    # --- Index bounds check ---
    if start_idx >= len(soc) or end_idx > len(soc) or start_idx >= end_idx:
        return None
    if num_points < 1:
        return None
    # --------------------------

    # --- Find end point based on zero current ---
    zero_current_indices = np.where(np.isclose(ibat[start_idx:end_idx], 0.0, atol=1e-6))[0]
    if len(zero_current_indices) > 0:
        candidate_end = start_idx + zero_current_indices[0]
        if candidate_end > start_idx + 2:
            end_point_idx = candidate_end
        else:
            end_point_idx = end_idx - 1
    else:
        end_point_idx = end_idx - 1
    # --------------------------------------------

    total_idx_span = end_point_idx - start_idx
    if total_idx_span < 1:
        return None

    slope = (soc_end - soc_start) / total_idx_span
    if abs(slope) < epsilon:
        return None  # no meaningful change in SoC

    # Evenly spaced target SoCs
    target_socs = np.linspace(soc_start, soc_end, num_points) if num_points > 1 else np.array([soc_start])

    for target_soc in target_socs:
        # Predict index from linear slope in index space
        idx_pred = int(round(start_idx + (target_soc - soc_start) / slope))

        # Clamp predicted index to valid range
        idx_pred = max(start_idx, min(end_point_idx, idx_pred))

        soc_measured = soc[idx_pred]

        if soc_reference_point is not None and abs(soc_measured - soc_reference_point) > 0.01:
            continue
        
        multiplier = 1.0
        if target_soc < 0.15:
            multiplier = 10
        elif target_soc > 0.85:
            multiplier = 10
        abs_errors.append(abs(multiplier*(soc_measured - target_soc)))

    if not abs_errors:
        return None

    cumulative_error = float(np.sum(abs_errors))
    mean_error = float(np.mean(abs_errors))
    normalized_cumulative_error = cumulative_error / len(abs_errors)

    return {
        "cumulative_error": cumulative_error,
        "normalized_cumulative_error": normalized_cumulative_error,
        "mean_error": mean_error,
        "soc_start": soc_start,
        "soc_end": soc_end,
        "t_start": float(time[start_idx]),      
        "t_stop": float(time[end_point_idx]),   
    }


def summarize_simulation_errors(
    output_dir: Path,
    summary_filename="errors_summary_total.csv",
):
    summary_data = []
    file_pattern = re.compile(r"error_results_R-([\d.]+)_Q-([\d.]+)\.csv")

    print(f"Scanning output directory: {output_dir}", flush=True)

    csv_files = sorted(output_dir.glob("error_results_R-*.csv"))
    total_files = len(csv_files)
    processed_files = 0

    for file in csv_files:
        match = file_pattern.match(file.name)
        if not match:
            continue

        R_value = float(match.group(1))
        Q_value = float(match.group(2))
        processed_files += 1

        print(f"\n[{processed_files}/{total_files}] Processing file: {file.name} (R={R_value}, Q={Q_value})", flush=True)

        stats = {
            k: {"count": 0, "sum": 0.0, "sum_sq": 0.0, "max": -np.inf}
            for est in estimators_to_plot
            for k in [
                f"avg_{est}_slope_error",
                f"{est}_slope_error",
                f"{est}_end_soc_error",
                f"{est}_deviation_error",
                f"mean_{est}_deviation_error",
            ]
        }

        try:
            df = pd.read_csv(file)

            for k in stats.keys():
                if k in df.columns:
                    vals = df[k].dropna().to_numpy()
                    if len(vals) > 0:
                        abs_vals = np.abs(vals)
                        stats[k]["count"] += len(abs_vals)
                        stats[k]["sum"] += np.sum(abs_vals)
                        stats[k]["sum_sq"] += np.sum(abs_vals**2)
                        stats[k]["max"] = max(stats[k]["max"], np.max(abs_vals))

        except Exception as e:
            print(f"  Skipping {file.name} due to error: {e}", flush=True)
            continue

        # Compute final stats
        row = {"R_value": R_value, "Q_value": Q_value}
        for k, st in stats.items():
            row[f"avg_{k}"] = (st["sum"] / st["count"]) if st["count"] > 0 else None
            row[f"rms_{k}"] = (st["sum_sq"] / st["count"])**0.5 if st["count"] > 0 else None
            row[f"max_{k}"] = st["max"] if st["count"] > 0 else None
            row[f"cumulative_{k}"] = st["sum"] if st["count"] > 0 else None
            row[f"count_{k}"] = st["count"]

        summary_data.append(row)
        print(f"  Finished R={R_value}, Q={Q_value}", flush=True)

    # Save summary
    summary_df = pd.DataFrame(summary_data).sort_values(by=["R_value", "Q_value"])
    summary_path = output_dir / summary_filename
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}", flush=True)

    # ---- Combined EKF metric (2×SOC + 1×Slope) ---- =
    for est in estimators_to_plot:
        slope_col = f"avg_{est}_slope_error"
        soc_col = f"avg_{est}_end_soc_error"
        if slope_col in summary_df.columns and soc_col in summary_df.columns:
            slope_max = summary_df[slope_col].max()
            soc_max = summary_df[soc_col].max()
            if not np.isnan(slope_max) and not np.isnan(soc_max):
                slope_norm = summary_df[slope_col] / slope_max
                soc_norm = summary_df[soc_col] / soc_max
                summary_df[f"Combined error weights - {est}"] = (2 * soc_norm + slope_norm) / 3
                summary_df[f"Combined cumulative error weights - {est}"] = 2*summary_df[f"cumulative_{est}_end_soc_error"] + summary_df[f"cumulative_{est}_slope_error"]
                print(f"  ✅ Added combined error metric for {est}", flush=True)

    summary_csv_path = output_dir / summary_filename
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n✅ Summary saved to: {summary_csv_path}", flush=True)

    correlation = summary_df[["cumulative_ekf1_slope_error", "cumulative_ekf1_end_soc_error"]].corr()
    print(f"\nCorrelation between cumulative slope error and cumulative avg end SOC error:\n{correlation}", flush=True)

    correlation = summary_df[["avg_mean_ekf1_deviation_error", "cumulative_ekf1_deviation_error"]].corr()
    print(f"\nCorrelation between avg deviation error and cumulative avg end SOC error:\n{correlation}", flush=True)

    correlation = summary_df[["avg_avg_ekf1_slope_error", "cumulative_ekf1_slope_error"]].corr()
    print(f"\nCorrelation between avg slope error and cumulative avg end SOC error:\n{correlation}", flush=True)

    correlation = summary_df[["avg_ekf1_end_soc_error", "cumulative_ekf1_end_soc_error"]].corr()
    print(f"\nCorrelation between avg end SOC error and cumulative avg end SOC error:\n{correlation}", flush=True)

    # --- Scatter and Surface Plots for Each EKF ---
    for est in estimators_to_plot:
        print(f"\nProcessing plots for {est}...", flush=True)

        # ---- Scatter Plots ----
        scatter_metrics = [
            #f"avg_{est}_end_soc_error",
            f"avg_avg_{est}_slope_error",
            f"avg_mean_{est}_deviation_error",
            f"Combined error weights - {est}",
            f"Combined cumulative error weights - {est}",
            f"cumulative_{est}_end_soc_error",
            f"cumulative_{est}_slope_error",
            f"cumulative_{est}_deviation_error",
        ]

        for col in scatter_metrics:
            if col not in summary_df.columns:
                print(f"  Skipping missing column: {col}")
                continue
            vmin, vmax = summary_df[col].min(), summary_df[col].max()
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(summary_df["R_value"], summary_df["Q_value"],
                                c=summary_df[col], cmap="viridis_r",
                                vmin=vmin, vmax=vmax)
            plt.colorbar(scatter, label=col)
            plt.xlabel("R_value")
            plt.ylabel("Q_value")
            plt.title(f"{col} across R and Q")
            plt.grid(True)
            plt.savefig(summary_csv_path.with_name(f"{col}.png"))
            plt.close()
            print(f"  Saved scatter plot for {col}", flush=True)

        # ---- Surface Plots ----
        surface_metrics = [
            f"Combined error weights - {est}",
            f"cumulative_{est}_deviation_error",
            f"Combined cumulative error weights - {est}",
        ]

        for col in surface_metrics:
            if col not in summary_df.columns:
                print(f"  Skipping missing column: {col}")
                continue

            df_valid = summary_df.dropna(subset=["R_value", "Q_value", col])
            if df_valid.empty:
                print(f"  No valid data for {col}")
                continue

            R = df_valid["R_value"].values
            Q = df_valid["Q_value"].values
            Z = df_valid[col].values
            Z_inverted = -Z  # invert so that lowest error is highest point

            min_idx = np.argmin(Z)
            R_min, Q_min, Z_min = R[min_idx], Q[min_idx], Z_inverted[min_idx]

            fig = go.Figure(data=[
                go.Mesh3d(
                    x=R, y=Q, z=Z_inverted,
                    intensity=Z_inverted,
                    colorscale='Viridis',
                    opacity=0.8,
                    hovertemplate=f'R_value: %{{x}}<br>Q_value: %{{y}}<br>{col}: %{{z}}<extra></extra>'
                ),
                go.Scatter3d(
                    x=[R_min], y=[Q_min], z=[Z_min],
                    mode='markers',
                    marker=dict(size=6, color='red'),
                    name='Lowest error'
                )
            ])

            fig.update_layout(
                scene=dict(
                    xaxis_title='R_value',
                    yaxis_title='Q_value',
                    zaxis_title=f"Inverted {col}"
                ),
                title=f"Interactive Surface plot: {col} (inverted, lowest error highlighted)",
                margin=dict(l=0, r=0, t=50, b=0)
            )
            fig.show()






















