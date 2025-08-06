#!/usr/bin/env python3
"""
Battery Libraries Generator

Generates C header and implementation files for:
1. Battery data (battery_data_<name>.h) - Contains characterized battery parameters
2. Battery model (battery_model.h/c) - Generic implementation of BatteryModel functions
3. EKF fuel gauge (battery_ekf.h/c) - Generic implementation of EkfEstimator functions

Usage:
    python generate_battery_libraries.py --output-dir=<directory> [--battery-name=<name>]
"""

import os
import argparse
import json
import numpy as np
from pathlib import Path

def generate_battery_data_header(battery_model_data, battery_name, output_path):
    """
    Generate C header file containing battery data lookup tables and parameters

    Parameters:
    - battery_model_data: Dictionary containing the complete battery model data
    - battery_name: String identifier for the battery model
    - output_path: Path to save the header file
    """
    header_guard = f"BATTERY_DATA_{battery_name.upper()}_H"

    # Extract temperature points from the battery model data
    # Get separate temperature arrays for charging and discharging
    temp_keys = sorted(list(battery_model_data['ocv_curves'].keys()))
    temp_points_dischg = []
    temp_points_chg = []
    for key in temp_keys:
        temp_data = battery_model_data['ocv_curves'][key]
        temp_points_dischg.append(temp_data['bat_temp_dischg'])
        temp_points_chg.append(temp_data['bat_temp_chg'])
    
    # Sort both temperature arrays
    temp_points_dischg = sorted(temp_points_dischg)
    temp_points_chg = sorted(temp_points_chg)

    header = [
        "/**",
        f" * Battery Data: {battery_name.upper()}",
        " * Auto-generated from battery characterization data",
        " * Contains lookup tables and parameters for the specific battery model",
        " */",
        "",
        f"#ifndef {header_guard}",
        f"#define {header_guard}",
        "",
        "#include <stdint.h>",
        "",
        "/**",
        f" * Battery Specifications:",
        f" * Model: {battery_name.upper()}",
        " * Chemistry: LiFePO4",
        " * Characterized on: TODO - Add date",
        " */",
        "",
        "// Configuration",
        f"#define BATTERY_NUM_TEMP_POINTS {len(temp_keys)}",
        "",
        "// SOC breakpoints for piecewise functions",
        "#define BATTERY_SOC_BREAKPOINT_1 0.25f",
        "#define BATTERY_SOC_BREAKPOINT_2 0.8f",
        "",
        "// Temperature points arrays (in Celsius)",
        "// Discharge temperatures",
        f"static const float BATTERY_TEMP_POINTS_DISCHG[BATTERY_NUM_TEMP_POINTS] = {{",
    ]

    # Add discharge temperature points
    temp_strings_dischg = [f"    {temp:.2f}f" for temp in temp_points_dischg]
    header.append(", ".join(temp_strings_dischg))
    header.append("};")
    header.append("")
    
    # Add charge temperature points  
    header.append("// Charge temperatures")
    header.append(f"static const float BATTERY_TEMP_POINTS_CHG[BATTERY_NUM_TEMP_POINTS] = {{")
    temp_strings_chg = [f"    {temp:.2f}f" for temp in temp_points_chg]
    header.append(", ".join(temp_strings_chg))
    header.append("};")
    header.append("")

    # Add internal resistance parameters
    header.append("// Internal resistance curve parameters (rational function parameters a+b*t)/(c+d*t)")
    header.append("static const float BATTERY_R_INT_PARAMS[4] = {")
    r_int_params = battery_model_data['r_int']
    header.append(f"    // a, b, c, d for rational function (a + b*t)/(c + d*t)")
    header.append(f"    {r_int_params[0]:.6f}f, {r_int_params[1]:.6f}f, {r_int_params[2]:.6f}f, {r_int_params[3]:.6f}f")
    header.append("};")
    header.append("")

    # Add discharge OCV curve parameters for each temperature
    header.append("// Discharge OCV curve parameters for each temperature")
    header.append("static const float BATTERY_OCV_DISCHARGE_PARAMS[BATTERY_NUM_TEMP_POINTS][10] = {")

    for temp_idx, temp_key in enumerate(temp_keys):
        ocv_data = battery_model_data['ocv_curves'][temp_key]
        ocv_params = ocv_data['ocv_dischg']  # Updated key name
        actual_temp = ocv_data['bat_temp_dischg']

        header.append(f"    // Temperature: {actual_temp:.2f}°C (key: {temp_key})")
        header.append("    {")
        header.append(f"        {ocv_params[0]:.6f}f, {ocv_params[1]:.6f}f, // m, b (linear segment)")
        header.append(f"        {ocv_params[2]:.6f}f, {ocv_params[3]:.6f}f, {ocv_params[4]:.6f}f, {ocv_params[5]:.6f}f, // a1, b1, c1, d1 (first rational segment)")
        header.append(f"        {ocv_params[6]:.6f}f, {ocv_params[7]:.6f}f, {ocv_params[8]:.6f}f, {ocv_params[9]:.6f}f  // a3, b3, c3, d3 (third rational segment)")
        header.append("    }" + ("," if temp_idx < len(temp_keys)-1 else ""))

    header.append("};")
    header.append("")

    # Add charge OCV curve parameters for each temperature
    header.append("// Charge OCV curve parameters for each temperature")
    header.append("static const float BATTERY_OCV_CHARGE_PARAMS[BATTERY_NUM_TEMP_POINTS][10] = {")

    for temp_idx, temp_key in enumerate(temp_keys):
        ocv_data = battery_model_data['ocv_curves'][temp_key]
        ocv_params = ocv_data['ocv_chg']  # Updated key name
        actual_temp = ocv_data['bat_temp_chg']

        header.append(f"    // Temperature: {actual_temp:.2f}°C (key: {temp_key})")
        header.append("    {")
        header.append(f"        {ocv_params[0]:.6f}f, {ocv_params[1]:.6f}f, // m, b (linear segment)")
        header.append(f"        {ocv_params[2]:.6f}f, {ocv_params[3]:.6f}f, {ocv_params[4]:.6f}f, {ocv_params[5]:.6f}f, // a1, b1, c1, d1 (first rational segment)")
        header.append(f"        {ocv_params[6]:.6f}f, {ocv_params[7]:.6f}f, {ocv_params[8]:.6f}f, {ocv_params[9]:.6f}f  // a3, b3, c3, d3 (third rational segment)")
        header.append("    }" + ("," if temp_idx < len(temp_keys)-1 else ""))

    header.append("};")
    header.append("")

    # Add capacity data
    header.append("// Battery capacity data for each temperature")
    header.append("static const float BATTERY_CAPACITY[BATTERY_NUM_TEMP_POINTS][2] = {")

    for temp_idx, temp_key in enumerate(temp_keys):
        ocv_data = battery_model_data['ocv_curves'][temp_key]
        discharge_capacity = ocv_data['total_capacity_dischg']  # Updated key name
        charge_capacity = ocv_data['total_capacity_chg']  # Updated key name
        actual_temp = ocv_data['bat_temp_dischg']

        header.append(f"    // Temperature: {actual_temp:.2f}°C (key: {temp_key})")
        header.append(f"    {{ {discharge_capacity:.2f}f, {charge_capacity:.2f}f }}" + ("," if temp_idx < len(temp_keys)-1 else ""))

    header.append("};")
    header.append("")
    header.append(f"#endif // {header_guard}")

    # Write the header to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(header))

    print(f"Battery data header file generated: {output_path}")

def generate_battery_model_header(output_path):
    """
    Generate C header file for battery model functions

    Parameters:
    - output_path: Path to save the header file
    """
    header_guard = "BATTERY_MODEL_H"

    header = [
        "/**",
        " * Battery Model Interface",
        " * Auto-generated from BatteryModel Python class",
        " */",
        "",
        f"#ifndef {header_guard}",
        f"#define {header_guard}",
        "",
        "#include <stdint.h>",
        "#include <stdbool.h>",
        "",
        "// Include the battery data header - this will be selected at compile time",
        "// based on which battery is being used",
        "#include \"battery_data.h\"",
        "",
        "/**",
        " * Calculate internal resistance at the given temperature",
        " * @param temperature Battery temperature in Celsius",
        " * @return Internal resistance in ohms",
        " */",
        "float battery_rint(float temperature);",
        "",
        "/**",
        " * Get battery total capacity at the given temperature and discharge mode",
        " * @param temperature Battery temperature in Celsius",
        " * @param discharging_mode true if discharging, false if charging",
        " * @return Total capacity in mAh",
        " */",
        "float battery_total_capacity(float temperature, bool discharging_mode);",
        "",
        "/**",
        " * Calculate OCV from measured voltage and current",
        " * @param voltage_V Measured battery voltage in volts",
        " * @param current_mA Measured battery current in mA (positive for discharge)",
        " * @param temperature Battery temperature in Celsius",
        " * @return Open circuit voltage (OCV) in volts",
        " */",
        "float battery_meas_to_ocv(float voltage_V, float current_mA, float temperature);",
        "",
        "/**",
        " * Get OCV for given SOC and temperature",
        " * @param soc State of charge (0.0 to 1.0)",
        " * @param temperature Battery temperature in Celsius",
        " * @param discharging_mode true if discharging, false if charging",
        " * @return Open circuit voltage in volts",
        " */",
        "float battery_ocv(float soc, float temperature, bool discharging_mode);",
        "",
        "/**",
        " * Get the slope of the OCV curve at a given SOC and temperature",
        " * @param soc State of charge (0.0 to 1.0)",
        " * @param temperature Battery temperature in Celsius",
        " * @param discharging_mode true if discharging, false if charging",
        " * @return Slope of OCV curve (dOCV/dSOC) in volts",
        " */",
        "float battery_ocv_slope(float soc, float temperature, bool discharging_mode);",
        "",
        "/**",
        " * Get SOC for given OCV and temperature",
        " * @param ocv Open circuit voltage in volts",
        " * @param temperature Battery temperature in Celsius",
        " * @param discharging_mode true if discharging, false if charging",
        " * @return State of charge (0.0 to 1.0)",
        " */",
        "float battery_soc(float ocv, float temperature, bool discharging_mode);",
        "",
        f"#endif // {header_guard}"
    ]

    # Write the header to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(header))

    print(f"Battery model header file generated: {output_path}")

def generate_battery_model_implementation(output_path):
    """
    Generate C implementation file for battery model functions

    Parameters:
    - output_path: Path to save the implementation file
    """
    impl = [
        "/**",
        " * Battery Model Implementation",
        " * Auto-generated from BatteryModel Python class",
        " */",
        "",
        "#include \"battery_model.h\"",
        "#include <math.h>",
        "",
        "// Helper function for linear interpolation",
        "static float linear_interpolate(float x, float x1, float y1, float x2, float y2) {",
        "    // Prevent division by zero",
        "    if (fabsf(x2 - x1) < 1e-6f) {",
        "        return (y1 + y2) / 2.0f;  // Return average if x values are too close",
        "    }",
        "    return y1 + (x - x1) * (y2 - y1) / (x2 - x1);",
        "}",
        "",
        "// Calculate OCV for specific parameters and SOC",
        "static float calc_ocv(const float* params, float soc) {",
        "    if (soc < BATTERY_SOC_BREAKPOINT_1) {",
        "        // First segment (rational function): (a1 + b1*x)/(c1 + d1*x)",
        "        float a1 = params[2];",
        "        float b1 = params[3];",
        "        float c1 = params[4];",
        "        float d1 = params[5];",
        "        return (a1 + b1 * soc) / (c1 + d1 * soc);",
        "    } ",
        "    else if (soc <= BATTERY_SOC_BREAKPOINT_2) {",
        "        // Middle segment (linear function): m*x + b",
        "        float m = params[0];",
        "        float b = params[1];",
        "        return m * soc + b;",
        "    } ",
        "    else {",
        "        // Third segment (rational function): (a3 + b3*x)/(c3 + d3*x)",
        "        float a3 = params[6];",
        "        float b3 = params[7];",
        "        float c3 = params[8];",
        "        float d3 = params[9];",
        "        return (a3 + b3 * soc) / (c3 + d3 * soc);",
        "    }",
        "}",
        "",
        "// Calculate OCV slope for specific parameters and SOC",
        "static float calc_ocv_slope(const float* params, float soc) {",
        "    if (soc < BATTERY_SOC_BREAKPOINT_1) {",
        "        // First segment (rational function derivative)",
        "        float a1 = params[2];",
        "        float b1 = params[3];",
        "        float c1 = params[4];",
        "        float d1 = params[5];",
        "        float denominator = c1 + d1 * soc;",
        "        return (b1 * c1 - a1 * d1) / (denominator * denominator);",
        "    } ",
        "    else if (soc <= BATTERY_SOC_BREAKPOINT_2) {",
        "        // Middle segment (linear function derivative)",
        "        float m = params[0];",
        "        return m;",
        "    } ",
        "    else {",
        "        // Third segment (rational function derivative)",
        "        float a3 = params[6];",
        "        float b3 = params[7];",
        "        float c3 = params[8];",
        "        float d3 = params[9];",
        "        float denominator = c3 + d3 * soc;",
        "        return (b3 * c3 - a3 * d3) / (denominator * denominator);",
        "    }",
        "}",
        "",
        "// Calculate SOC from OCV for specific parameters",
        "static float calc_soc_from_ocv(const float* params, float ocv) {",
        "    // Calculate breakpoint voltages",
        "    float ocv_breakpoint_1 = calc_ocv(params, BATTERY_SOC_BREAKPOINT_1);",
        "    float ocv_breakpoint_2 = calc_ocv(params, BATTERY_SOC_BREAKPOINT_2);",
        "    ",
        "    // Extract parameters",
        "    float m = params[0];",
        "    float b = params[1];",
        "    float a1 = params[2];",
        "    float b1 = params[3];",
        "    float c1 = params[4];",
        "    float d1 = params[5];",
        "    float a3 = params[6];",
        "    float b3 = params[7];",
        "    float c3 = params[8];",
        "    float d3 = params[9];",
        "    ",
        "    if (ocv < ocv_breakpoint_1) {",
        "        // First segment (rational function inverse)",
        "        return (a1 - c1 * ocv) / (d1 * ocv - b1);",
        "    } ",
        "    else if (ocv <= ocv_breakpoint_2) {",
        "        // Middle segment (linear function inverse)",
        "        return (ocv - b) / m;",
        "    } ",
        "    else {",
        "        // Third segment (rational function inverse)",
        "        return (a3 - c3 * ocv) / (d3 * ocv - b3);",
        "    }",
        "}",
        "",
        "float battery_rint(float temperature) {",
        "    // Calculate R_int using rational function: (a + b*t)/(c + d*t)",
        "    float a = BATTERY_R_INT_PARAMS[0];",
        "    float b = BATTERY_R_INT_PARAMS[1];",
        "    float c = BATTERY_R_INT_PARAMS[2];",
        "    float d = BATTERY_R_INT_PARAMS[3];",
        "    ",
        "    return (a + b * temperature) / (c + d * temperature);",
        "}",
        "",
        "float battery_total_capacity(float temperature, bool discharging_mode) {",
        "    // Select appropriate temperature array based on mode",
        "    const float* temp_points = discharging_mode ? BATTERY_TEMP_POINTS_DISCHG : BATTERY_TEMP_POINTS_CHG;",
        "    ",
        "    // Handle out-of-bounds temperatures",
        "    if (temperature <= temp_points[0]) {",
        "        return BATTERY_CAPACITY[0][discharging_mode ? 0 : 1];",
        "    }",
        "    ",
        "    if (temperature >= temp_points[BATTERY_NUM_TEMP_POINTS - 1]) {",
        "        return BATTERY_CAPACITY[BATTERY_NUM_TEMP_POINTS - 1][discharging_mode ? 0 : 1];",
        "    }",
        "    ",
        "    // Find temperature bracket",
        "    for (int i = 0; i < BATTERY_NUM_TEMP_POINTS - 1; i++) {",
        "        if (temperature < temp_points[i + 1]) {",
        "            return linear_interpolate(temperature,",
        "                                     temp_points[i], BATTERY_CAPACITY[i][discharging_mode ? 0 : 1],",
        "                                     temp_points[i + 1], BATTERY_CAPACITY[i + 1][discharging_mode ? 0 : 1]);",
        "        }",
        "    }",
        "    ",
        "    // Should never reach here",
        "    return BATTERY_CAPACITY[0][discharging_mode ? 0 : 1];",
        "}",
        "",
        "float battery_meas_to_ocv(float voltage_V, float current_mA, float temperature) {",
        "    // Convert mA to A by dividing by 1000",
        "    float current_A = current_mA / 1000.0f;",
        "    ",
        "    // Calculate OCV: V_OC = V_term + I * R_int",
        "    return voltage_V + (current_A * battery_rint(temperature));",
        "}",
        "",
        "float battery_ocv(float soc, float temperature, bool discharging_mode) {",
        "    // Clamp SOC to valid range",
        "    soc = (soc < 0.0f) ? 0.0f : ((soc > 1.0f) ? 1.0f : soc);",
        "    ",
        "    // Select appropriate temperature array based on mode",
        "    const float* temp_points = discharging_mode ? BATTERY_TEMP_POINTS_DISCHG : BATTERY_TEMP_POINTS_CHG;",
        "    ",
        "    // Handle out-of-bounds temperatures",
        "    if (temperature <= temp_points[0]) {",
        "        const float* params = discharging_mode ? ",
        "                            BATTERY_OCV_DISCHARGE_PARAMS[0] : ",
        "                            BATTERY_OCV_CHARGE_PARAMS[0];",
        "        return calc_ocv(params, soc);",
        "    }",
        "    ",
        "    if (temperature >= temp_points[BATTERY_NUM_TEMP_POINTS - 1]) {",
        "        const float* params = discharging_mode ? ",
        "                            BATTERY_OCV_DISCHARGE_PARAMS[BATTERY_NUM_TEMP_POINTS - 1] : ",
        "                            BATTERY_OCV_CHARGE_PARAMS[BATTERY_NUM_TEMP_POINTS - 1];",
        "        return calc_ocv(params, soc);",
        "    }",
        "    ",
        "    // Find temperature bracket and interpolate",
        "    for (int i = 0; i < BATTERY_NUM_TEMP_POINTS - 1; i++) {",
        "        if (temperature < temp_points[i + 1]) {",
        "            const float* params_low = discharging_mode ? ",
        "                                    BATTERY_OCV_DISCHARGE_PARAMS[i] : ",
        "                                    BATTERY_OCV_CHARGE_PARAMS[i];",
        "            ",
        "            const float* params_high = discharging_mode ? ",
        "                                     BATTERY_OCV_DISCHARGE_PARAMS[i + 1] : ",
        "                                     BATTERY_OCV_CHARGE_PARAMS[i + 1];",
        "            ",
        "            float ocv_low = calc_ocv(params_low, soc);",
        "            float ocv_high = calc_ocv(params_high, soc);",
        "            ",
        "            return linear_interpolate(temperature,",
        "                                     temp_points[i], ocv_low,",
        "                                     temp_points[i + 1], ocv_high);",
        "        }",
        "    }",
        "    ",
        "    // Should never reach here",
        "    const float* params = discharging_mode ? ",
        "                        BATTERY_OCV_DISCHARGE_PARAMS[0] : ",
        "                        BATTERY_OCV_CHARGE_PARAMS[0];",
        "    return calc_ocv(params, soc);",
        "}",
        "",
        "float battery_ocv_slope(float soc, float temperature, bool discharging_mode) {",
        "    // Clamp SOC to valid range",
        "    soc = (soc < 0.0f) ? 0.0f : ((soc > 1.0f) ? 1.0f : soc);",
        "    ",
        "    // Select appropriate temperature array based on mode",
        "    const float* temp_points = discharging_mode ? BATTERY_TEMP_POINTS_DISCHG : BATTERY_TEMP_POINTS_CHG;",
        "    ",
        "    // Handle out-of-bounds temperatures",
        "    if (temperature <= temp_points[0]) {",
        "        const float* params = discharging_mode ? ",
        "                            BATTERY_OCV_DISCHARGE_PARAMS[0] : ",
        "                            BATTERY_OCV_CHARGE_PARAMS[0];",
        "        return calc_ocv_slope(params, soc);",
        "    }",
        "    ",
        "    if (temperature >= temp_points[BATTERY_NUM_TEMP_POINTS - 1]) {",
        "        const float* params = discharging_mode ? ",
        "                            BATTERY_OCV_DISCHARGE_PARAMS[BATTERY_NUM_TEMP_POINTS - 1] : ",
        "                            BATTERY_OCV_CHARGE_PARAMS[BATTERY_NUM_TEMP_POINTS - 1];",
        "        return calc_ocv_slope(params, soc);",
        "    }",
        "    ",
        "    // Find temperature bracket and interpolate",
        "    for (int i = 0; i < BATTERY_NUM_TEMP_POINTS - 1; i++) {",
        "        if (temperature < temp_points[i + 1]) {",
        "            const float* params_low = discharging_mode ? ",
        "                                    BATTERY_OCV_DISCHARGE_PARAMS[i] : ",
        "                                    BATTERY_OCV_CHARGE_PARAMS[i];",
        "            ",
        "            const float* params_high = discharging_mode ? ",
        "                                     BATTERY_OCV_DISCHARGE_PARAMS[i + 1] : ",
        "                                     BATTERY_OCV_CHARGE_PARAMS[i + 1];",
        "            ",
        "            float slope_low = calc_ocv_slope(params_low, soc);",
        "            float slope_high = calc_ocv_slope(params_high, soc);",
        "            ",
        "            return linear_interpolate(temperature,",
        "                                     temp_points[i], slope_low,",
        "                                     temp_points[i + 1], slope_high);",
        "        }",
        "    }",
        "    ",
        "    // Should never reach here",
        "    const float* params = discharging_mode ? ",
        "                        BATTERY_OCV_DISCHARGE_PARAMS[0] : ",
        "                        BATTERY_OCV_CHARGE_PARAMS[0];",
        "    return calc_ocv_slope(params, soc);",
        "}",
        "",
        "float battery_soc(float ocv, float temperature, bool discharging_mode) {",
        "    // Select appropriate temperature array based on mode",
        "    const float* temp_points = discharging_mode ? BATTERY_TEMP_POINTS_DISCHG : BATTERY_TEMP_POINTS_CHG;",
        "    ",
        "    // Handle out-of-bounds temperatures",
        "    if (temperature <= temp_points[0]) {",
        "        const float* params = discharging_mode ? ",
        "                            BATTERY_OCV_DISCHARGE_PARAMS[0] : ",
        "                            BATTERY_OCV_CHARGE_PARAMS[0];",
        "        return calc_soc_from_ocv(params, ocv);",
        "    }",
        "    ",
        "    if (temperature >= temp_points[BATTERY_NUM_TEMP_POINTS - 1]) {",
        "        const float* params = discharging_mode ? ",
        "                            BATTERY_OCV_DISCHARGE_PARAMS[BATTERY_NUM_TEMP_POINTS - 1] : ",
        "                            BATTERY_OCV_CHARGE_PARAMS[BATTERY_NUM_TEMP_POINTS - 1];",
        "        return calc_soc_from_ocv(params, ocv);",
        "    }",
        "    ",
        "    // Find temperature bracket and interpolate",
        "    for (int i = 0; i < BATTERY_NUM_TEMP_POINTS - 1; i++) {",
        "        if (temperature < temp_points[i + 1]) {",
        "            const float* params_low = discharging_mode ? ",
        "                                    BATTERY_OCV_DISCHARGE_PARAMS[i] : ",
        "                                    BATTERY_OCV_CHARGE_PARAMS[i];",
        "            ",
        "            const float* params_high = discharging_mode ? ",
        "                                     BATTERY_OCV_DISCHARGE_PARAMS[i + 1] : ",
        "                                     BATTERY_OCV_CHARGE_PARAMS[i + 1];",
        "            ",
        "            float soc_low = calc_soc_from_ocv(params_low, ocv);",
        "            float soc_high = calc_soc_from_ocv(params_high, ocv);",
        "            ",
        "            return linear_interpolate(temperature,",
        "                                     temp_points[i], soc_low,",
        "                                     temp_points[i + 1], soc_high);",
        "        }",
        "    }",
        "    ",
        "    // Should never reach here",
        "    const float* params = discharging_mode ? ",
        "                        BATTERY_OCV_DISCHARGE_PARAMS[0] : ",
        "                        BATTERY_OCV_CHARGE_PARAMS[0];",
        "    return calc_soc_from_ocv(params, ocv);",
        "}"
    ]

    # Write the implementation to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(impl))

    print(f"Battery model implementation file generated: {output_path}")

def generate_ekf_header(output_path):
    """
    Generate C header file for the EKF fuel gauge functions

    Parameters:
    - output_path: Path to save the header file
    """
    header_guard = "BATTERY_EKF_H"

    header = [
        "/**",
        " * Battery Extended Kalman Filter (EKF) Fuel Gauge",
        " * Auto-generated from EkfEstimator Python class",
        " */",
        "",
        f"#ifndef {header_guard}",
        f"#define {header_guard}",
        "",
        "#include <stdint.h>",
        "#include <stdbool.h>",
        "",
        "// EKF State structure",
        "typedef struct {",
        "    // State estimate (SOC)",
        "    float soc;",
        "    // Latched SOC (the one that gets reported)",
        "    float soc_latched;",
        "    // Error covariance",
        "    float P;",
        "    // Filter parameters",
        "    float R;         // Measurement noise variance",
        "    float Q;         // Process noise variance",
        "    float R_aggressive;  // Aggressive measurement noise variance",
        "    float Q_aggressive;  // Aggressive process noise variance",
        "    // Measurement history for filtering",
        "    float v_history[5];  // Voltage history buffer",
        "    float i_history[5];  // Current history buffer",
        "    uint8_t history_count; // Number of samples in history",
        "    uint8_t history_index; // Current index in circular buffer",
        "} battery_ekf_state_t;",
        "",
        "/**",
        " * Initialize the EKF state",
        " * @param state Pointer to EKF state structure",
        " * @param R Measurement noise variance",
        " * @param Q Process noise variance",
        " * @param R_aggressive Aggressive mode measurement noise variance",
        " * @param Q_aggressive Aggressive mode process noise variance",
        " * @param P_init Initial error covariance",
        " */",
        "void battery_ekf_init(battery_ekf_state_t* state, float R, float Q,",
        "                      float R_aggressive, float Q_aggressive, float P_init);",
        "",
        "/**",
        " * Reset the EKF state",
        " * @param state Pointer to EKF state structure",
        " */",
        "void battery_ekf_reset(battery_ekf_state_t* state);",
        "",
        "/**",
        " * Make initial SOC guess based on OCV",
        " * @param state Pointer to EKF state structure",
        " * @param voltage_V Current battery voltage (V)",
        " * @param current_mA Current battery current (mA), positive for discharge",
        " * @param temperature Battery temperature (°C)",
        " */",
        "void battery_ekf_initial_guess(battery_ekf_state_t* state,",
        "                               float voltage_V, float current_mA, float temperature);",
        "",
        "/**",
        " * Update the EKF with new measurements",
        " * @param state Pointer to EKF state structure",
        " * @param dt Time step in milliseconds",
        " * @param voltage_V Current battery voltage (V)",
        " * @param current_mA Current battery current (mA), positive for discharge",
        " * @param temperature Battery temperature (°C)",
        " * @return Updated SOC estimate (0.0 to 1.0)",
        " */",
        "float battery_ekf_update(battery_ekf_state_t* state, uint32_t dt,",
        "                         float voltage_V, float current_mA, float temperature);",
        "",
        f"#endif // {header_guard}"
    ]

    # Write the header to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(header))

    print(f"EKF header file generated: {output_path}")

def generate_ekf_implementation(output_path):
    """
    Generate C implementation file for EKF fuel gauge functions

    Parameters:
    - output_path: Path to save the implementation file
    """
    impl = [
        "/**",
        " * Battery Extended Kalman Filter (EKF) Fuel Gauge Implementation",
        " * Auto-generated from EkfEstimator Python class",
        " */",
        "",
        "#include \"battery_ekf.h\"",
        "#include \"battery_model.h\"",
        "#include <math.h>",
        "",
        "// Filter window size for measurements",
        "#define FILTER_WINDOW_SIZE 5",
        "",
        "// Helper function for filtered measurements",
        "static float filter_measurement(battery_ekf_state_t* state, float new_value, float* history) {",
        "    // Add new measurement to history",
        "    history[state->history_index] = new_value;",
        "    ",
        "    // Update index for circular buffer",
        "    state->history_index = (state->history_index + 1) % FILTER_WINDOW_SIZE;",
        "    ",
        "    // Update count for averaging",
        "    if (state->history_count < FILTER_WINDOW_SIZE) {",
        "        state->history_count++;",
        "    }",
        "    ",
        "    // Calculate average",
        "    float sum = 0.0f;",
        "    for (uint8_t i = 0; i < state->history_count; i++) {",
        "        sum += history[i];",
        "    }",
        "    ",
        "    return sum / state->history_count;",
        "}",
        "",
        "void battery_ekf_init(battery_ekf_state_t* state, float R, float Q,",
        "                      float R_aggressive, float Q_aggressive, float P_init) {",
        "    state->R = R;",
        "    state->Q = Q;",
        "    state->R_aggressive = R_aggressive;",
        "    state->Q_aggressive = Q_aggressive;",
        "    state->P = P_init;",
        "    ",
        "    // Initialize state",
        "    state->soc = 0.0f;",
        "    state->soc_latched = 0.0f;",
        "    state->history_count = 0;",
        "    state->history_index = 0;",
        "    ",
        "    // Initialize history arrays",
        "    for (int i = 0; i < FILTER_WINDOW_SIZE; i++) {",
        "        state->v_history[i] = 0.0f;",
        "        state->i_history[i] = 0.0f;",
        "    }",
        "}",
        "",
        "void battery_ekf_reset(battery_ekf_state_t* state) {",
        "    // Reset state but keep filter parameters",
        "    state->soc = 0.0f;",
        "    state->soc_latched = 0.0f;",
        "    state->history_count = 0;",
        "    state->history_index = 0;",
        "    ",
        "    // Reset history arrays",
        "    for (int i = 0; i < FILTER_WINDOW_SIZE; i++) {",
        "        state->v_history[i] = 0.0f;",
        "        state->i_history[i] = 0.0f;",
        "    }",
        "}",
        "",
        "void battery_ekf_initial_guess(battery_ekf_state_t* state,",
        "                               float voltage_V, float current_mA, float temperature) {",
        "    // Determine if we're in discharge mode",
        "    bool discharging_mode = current_mA >= 0.0f;",
        "",
        "    // Calculate OCV from terminal voltage and current",
        "    float ocv = battery_meas_to_ocv(voltage_V, current_mA, temperature);",
        "    ",
        "    // Get SOC from OCV using lookup",
        "    state->soc = battery_soc(ocv, temperature, discharging_mode);",
        "    state->soc_latched = state->soc;",
        "}",
        "",
        "float battery_ekf_update(battery_ekf_state_t* state, uint32_t dt,",
        "                         float voltage_V, float current_mA, float temperature) {",
        "    // Filter measurements",
        "    voltage_V = filter_measurement(state, voltage_V, state->v_history);",
        "    current_mA = filter_measurement(state, current_mA, state->i_history);",
        "    ",
        "    // Determine if we're in discharge mode",
        "    bool discharging_mode = current_mA >= 0.0f;",
        "    ",
        "    // Choose filter parameters based on temperature and SOC",
        "    float R = state->R;",
        "    float Q = state->Q;",
        "    ",
        "    if (temperature < 10.0f) {",
        "        // Cold temperature - use more conservative values",
        "        R = 10.0f;",
        "        Q = 0.01f;",
        "    } else if (state->soc_latched < 0.2f) {",
        "        // Low SOC - use aggressive values to track more closely",
        "        R = state->R_aggressive;",
        "        Q = state->Q_aggressive;",
        "    }",
        "    ",
        "    // Convert milliseconds to seconds",
        "    float dt_sec = dt / 1000.0f;",
        "    ",
        "    // Get total capacity at current temperature",
        "    float total_capacity = battery_total_capacity(temperature, discharging_mode);",
        "    ",
        "    // State prediction (coulomb counting)",
        "    // SOC_k+1 = SOC_k - (I*dt)/(3600*capacity)",
        "    float x_k1_k = state->soc - (current_mA / (3600.0f * total_capacity)) * dt_sec;",
        "    ",
        "    // Calculate Jacobian of measurement function h(x) = dOCV/dSOC",
        "    float h_jacobian = battery_ocv_slope(x_k1_k, temperature, discharging_mode);",
        "    ",
        "    // Error covariance prediction",
        "    float P_k1_k = state->P + Q;",
        "    ",
        "    // Calculate innovation covariance",
        "    float S = h_jacobian * P_k1_k * h_jacobian + R;",
        "    ",
        "    // Calculate Kalman gain",
        "    float K_k1_k = P_k1_k * h_jacobian / S;",
        "    ",
        "    // Calculate predicted terminal voltage",
        "    float v_pred = battery_ocv(x_k1_k, temperature, discharging_mode) - ",
        "                   (current_mA / 1000.0f) * battery_rint(temperature);",
        "    ",
        "    // State update",
        "    float x_k1_k1 = x_k1_k + K_k1_k * (voltage_V - v_pred);",
        "    ",
        "    // Error covariance update",
        "    float P_k1_k1 = (1.0f - K_k1_k * h_jacobian) * P_k1_k;",
        "    ",
        "    // Enforce SOC boundaries",
        "    state->soc = (x_k1_k1 < 0.0f) ? 0.0f : ((x_k1_k1 > 1.0f) ? 1.0f : x_k1_k1);",
        "    state->P = P_k1_k1;",
        "    ",
        "    // Update latched SOC based on current direction",
        "    if (current_mA > 0.0f) {",
        "        // Discharging, SOC should move only in negative direction",
        "        if (state->soc < state->soc_latched) {",
        "            state->soc_latched = state->soc;",
        "        }",
        "    } else {",
        "        // Charging, SOC should move only in positive direction",
        "        if (state->soc > state->soc_latched) {",
        "            state->soc_latched = state->soc;",
        "        }",
        "    }",
        "    ",
        "    return state->soc_latched;",
        "}"
    ]

    # Write the implementation to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(impl))

    print(f"EKF implementation file generated: {output_path}")

def generate_battery_data_symlink(battery_name, output_dir):
    """
    Create a symbolic link from battery_data.h to the specific battery data file

    Parameters:
    - battery_name: Name of the battery
    - output_dir: Output directory
    """
    src = f"battery_data_{battery_name.lower()}.h"
    dst = os.path.join(output_dir, "battery_data.h")

    # Create symbolic link (or copy file for Windows compatibility)
    if os.name == 'nt':  # Windows
        # Windows symlinks require admin privileges, so just copy the file
        import shutil
        shutil.copy2(os.path.join(output_dir, src), dst)
        print(f"Copied {src} to {dst}")
    else:  # Unix/Linux/Mac
        try:
            # Remove existing symlink if it exists
            if os.path.exists(dst) or os.path.islink(dst):
                os.unlink(dst)
            # Create new symlink
            os.symlink(src, dst)
            print(f"Created symlink from {src} to {dst}")
        except Exception as e:
            print(f"Error creating symlink: {e}")
            print(f"You may need to manually create a link/copy from {src} to {dst}")

def generate_battery_libraries(battery_model_data, output_dir=".", battery_name="default"):
    """
    Generate all battery-related libraries

    Parameters:
    - battery_model_data: Dictionary containing battery model data
    - output_dir: Directory to save the output files
    - battery_name: Name identifier for the battery
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define file paths
    data_h = os.path.join(output_dir, f"battery_data_{battery_name.lower()}.h")
    model_h = os.path.join(output_dir, "battery_model.h")
    model_c = os.path.join(output_dir, "battery_model.c")
    ekf_h = os.path.join(output_dir, "battery_ekf.h")
    ekf_c = os.path.join(output_dir, "battery_ekf.c")

    # Generate files
    generate_battery_data_header(battery_model_data, battery_name, data_h)
    generate_battery_model_header(model_h)
    generate_battery_model_implementation(model_c)
    generate_ekf_header(ekf_h)
    generate_ekf_implementation(ekf_c)

    # Create symlink or copy for battery_data.h
    generate_battery_data_symlink(battery_name, output_dir)

    print(f"All battery libraries generated in {output_dir}")

def generate_battery_readme(output_dir, battery_name):
    """Generate a README file explaining how to use the battery libraries"""
    readme_content = [
        "# Battery Model and Fuel Gauge Library",
        "",
        "This directory contains the generated battery model and fuel gauge libraries for embedded systems.",
        "",
        "## Files",
        "",
        f"- `battery_data_{battery_name.lower()}.h` - Specific battery data for {battery_name}",
        "- `battery_data.h` - Symbolic link to the active battery data file",
        "- `battery_model.h/.c` - Battery model implementation",
        "- `battery_ekf.h/.c` - Battery fuel gauge EKF implementation",
        "",
        "## How to Use",
        "",
        "1. Include the necessary headers in your application:",
        "   ```c",
        "   #include \"battery_model.h\"",
        "   #include \"battery_ekf.h\"",
        "   ```",
        "",
        "2. Initialize the EKF state:",
        "   ```c",
        "   battery_ekf_state_t ekf_state;",
        "   battery_ekf_init(&ekf_state, 0.01f, 0.001f, 0.005f, 0.005f, 0.1f);",
        "   ```",
        "",
        "3. Make an initial SOC guess based on voltage:",
        "   ```c",
        "   float voltage_V = 3.2f;    // Initial battery voltage",
        "   float current_mA = 0.0f;   // Initial current (assuming at rest)",
        "   float temperature = 25.0f; // Battery temperature in Celsius",
        "   battery_ekf_initial_guess(&ekf_state, voltage_V, current_mA, temperature);",
        "   ```",
        "",
        "4. Periodically update the SOC estimate:",
        "   ```c",
        "   uint32_t dt = 1000; // Time since last update in milliseconds",
        "   float soc = battery_ekf_update(&ekf_state, dt, voltage_V, current_mA, temperature);",
        "   ```",
        "",
        "## Changing Battery Models",
        "",
        "To use a different battery model:",
        "",
        "1. Generate a new battery data header file using the characterization tool.",
        "2. Replace the `battery_data.h` symlink to point to the new battery data file.",
        "   - Linux: `ln -sf battery_data_newbattery.h battery_data.h`",
        "   - Windows: Copy the new file to `battery_data.h`",
        "",
        "The API functions in `battery_model.h` and `battery_ekf.h` remain the same regardless of",
        "which battery data file is used.",
    ]

    # Write README.md
    with open(os.path.join(output_dir, "README.md"), 'w') as f:
        f.write('\n'.join(readme_content))

    print(f"README file generated in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate battery model and EKF libraries')
    parser.add_argument('--output-dir', default='generated_output', help='Output directory for generated files')
    parser.add_argument('--battery-name', default='lifepo4', help='Battery name identifier')

    args = parser.parse_args()

    # If this script is run directly, try to import the battery model data
    try:
        # Import battery_model_data from main script if available
        from archive.run_battery_characterization import battery_model_data
        generate_battery_libraries(battery_model_data, args.output_dir, args.battery_name)
        generate_battery_readme(args.output_dir, args.battery_name)
    except (ImportError, AttributeError):
        print("No battery_model_data found. Please run this script after battery characterization.")