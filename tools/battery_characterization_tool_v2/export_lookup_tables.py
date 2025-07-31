#!/usr/bin/env python3
"""
Battery Model Lookup Table Exporter

This script generates C header and implementation files for battery model data,
converting battery_model_data Python dictionary into C lookup tables and functions.
It implements the BatteryModel class functions in C code for embedded applications.
"""

import os
import json
import numpy as np
from pathlib import Path


def generate_battery_model_header(battery_model_data, output_path="battery_model.h"):
    """
    Generate C header file from the battery_model_data dictionary

    Parameters:
    - battery_model_data: Dictionary containing 'r_int' parameters and 'ocv_curves' data
    - output_path: Path to save the header file
    """
    # Get temperature points (sorted)
    temp_points = sorted(list(battery_model_data['ocv_curves'].keys()))
    num_temp_points = len(temp_points)

    # Start building the header file content
    header = [
        "/**",
        " * Battery Model Lookup Tables",
        " * Auto-generated from battery characterization data",
        " */",
        "",
        "#ifndef BATTERY_MODEL_H",
        "#define BATTERY_MODEL_H",
        "",
        "#include <stdint.h>",
        "#include <stdbool.h>",
        "",
        "// Configuration",
        f"#define BATTERY_NUM_TEMPERATURE_POINTS {num_temp_points}",
        "",
        "// Battery model parameters",
        "// SOC breakpoints for piecewise functions",
        "#define BATTERY_SOC_BREAKPOINT_1 0.25f",
        "#define BATTERY_SOC_BREAKPOINT_2 0.8f",
        "",
        "// Temperature points array (in Celsius)",
        "static const float BATTERY_TEMP_POINTS[BATTERY_NUM_TEMPERATURE_POINTS] = {",
        "    " + ", ".join([f"{temp:.2f}f" for temp in temp_points]),
        "};",
        "",
        "// Internal resistance curve parameters (rational function parameters a+b*t)/(c+d*t)",
        "typedef struct {",
        "    float a;",
        "    float b;",
        "    float c;",
        "    float d;",
        "} rint_params_t;",
        "",
        "// OCV curve parameters for one temperature",
        "typedef struct {",
        "    // m, b (linear segment)",
        "    float m;",
        "    float b;",
        "    // a1, b1, c1, d1 (first rational segment)",
        "    float a1;",
        "    float b1;",
        "    float c1;",
        "    float d1;",
        "    // a3, b3, c3, d3 (third rational segment)",
        "    float a3;",
        "    float b3;",
        "    float c3;",
        "    float d3;",
        "    // Total capacity at this temperature",
        "    float total_capacity;",
        "} ocv_params_t;",
        "",
    ]

    # Add internal resistance parameters
    r_int_params = battery_model_data['r_int']
    header.extend([
        "// Internal resistance curve parameters",
        "static const rint_params_t BATTERY_R_INT_PARAMS = {",
        f"    .a = {r_int_params[0]:.6f}f,",
        f"    .b = {r_int_params[1]:.6f}f,",
        f"    .c = {r_int_params[2]:.6f}f,",
        f"    .d = {r_int_params[3]:.6f}f",
        "};",
        ""
    ])

    # Add OCV curve parameters for each temperature
    header.append("// OCV curve parameters for each temperature")
    header.append("static const ocv_params_t BATTERY_OCV_PARAMS[BATTERY_NUM_TEMPERATURE_POINTS] = {")

    for temp_idx, temp in enumerate(temp_points):
        ocv_data = battery_model_data['ocv_curves'][temp]
        ocv_params = ocv_data['ocv']
        total_capacity = ocv_data['total_capacity']

        header.append(f"    // Temperature: {temp:.2f}°C")
        header.append("    {")
        header.append(f"        .m = {ocv_params[0]:.6f}f,")
        header.append(f"        .b = {ocv_params[1]:.6f}f,")
        header.append(f"        .a1 = {ocv_params[2]:.6f}f,")
        header.append(f"        .b1 = {ocv_params[3]:.6f}f,")
        header.append(f"        .c1 = {ocv_params[4]:.6f}f,")
        header.append(f"        .d1 = {ocv_params[5]:.6f}f,")
        header.append(f"        .a3 = {ocv_params[6]:.6f}f,")
        header.append(f"        .b3 = {ocv_params[7]:.6f}f,")
        header.append(f"        .c3 = {ocv_params[8]:.6f}f,")
        header.append(f"        .d3 = {ocv_params[9]:.6f}f,")
        header.append(f"        .total_capacity = {total_capacity:.2f}f")
        header.append("    }" + ("," if temp_idx < len(temp_points)-1 else ""))

    header.append("};")
    header.append("")

    # Add function declarations (matching the BatteryModel class methods)
    header.extend([
        "// Function declarations",
        "",
        "/**",
        " * Calculate internal resistance at the given temperature",
        " * @param temperature Battery temperature in Celsius",
        " * @return Internal resistance in ohms",
        " */",
        "float battery_rint(float temperature);",
        "",
        "/**",
        " * Get battery total capacity at the given temperature",
        " * @param temperature Battery temperature in Celsius",
        " * @return Total capacity in mAh",
        " */",
        "float battery_total_capacity(float temperature);",
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
        " * @return Open circuit voltage in volts",
        " */",
        "float battery_ocv(float soc, float temperature);",
        "",
        "/**",
        " * Get the slope of the OCV curve at a given SOC and temperature",
        " * @param soc State of charge (0.0 to 1.0)",
        " * @param temperature Battery temperature in Celsius",
        " * @return Slope of OCV curve (dOCV/dSOC) in volts",
        " */",
        "float battery_ocv_slope(float soc, float temperature);",
        "",
        "/**",
        " * Get SOC for given OCV and temperature",
        " * @param ocv Open circuit voltage in volts",
        " * @param temperature Battery temperature in Celsius",
        " * @return State of charge (0.0 to 1.0)",
        " */",
        "float battery_soc(float ocv, float temperature);",
        "",
        "#endif // BATTERY_MODEL_H"
    ])

    # Write the header to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(header))

    print(f"Battery model header file generated: {output_path}")


def generate_battery_model_implementation(battery_model_data, output_path="battery_model.c"):
    """
    Generate C implementation file for battery model functions

    Parameters:
    - battery_model_data: Dictionary containing 'r_int' parameters and 'ocv_curves' data
    - output_path: Path to save the implementation file
    """
    # Start building the implementation file
    impl = [
        "/**",
        " * Battery Model Implementation",
        " * Auto-generated from battery characterization data",
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
        "// Internal helper function to get an OCV curve for a specific temperature",
        "static const ocv_params_t* get_ocv_params_for_temp(float temperature) {",
        "    // Check temperature boundaries",
        "    if (temperature <= BATTERY_TEMP_POINTS[0]) {",
        "        return &BATTERY_OCV_PARAMS[0];",
        "    }",
        "    ",
        "    if (temperature >= BATTERY_TEMP_POINTS[BATTERY_NUM_TEMPERATURE_POINTS - 1]) {",
        "        return &BATTERY_OCV_PARAMS[BATTERY_NUM_TEMPERATURE_POINTS - 1];",
        "    }",
        "    ",
        "    // Find temperature bracket",
        "    for (int i = 0; i < BATTERY_NUM_TEMPERATURE_POINTS - 1; i++) {",
        "        if (temperature < BATTERY_TEMP_POINTS[i + 1]) {",
        "            return &BATTERY_OCV_PARAMS[i];",
        "        }",
        "    }",
        "    ",
        "    // Should never reach here",
        "    return &BATTERY_OCV_PARAMS[0];",
        "}",
        "",
        "// Helper function to calculate OCV for specific parameters and SOC",
        "static float calc_ocv(const ocv_params_t* params, float soc) {",
        "    if (soc < BATTERY_SOC_BREAKPOINT_1) {",
        "        // First segment (rational)",
        "        return (params->a1 + params->b1 * soc) / (params->c1 + params->d1 * soc);",
        "    } ",
        "    else if (soc <= BATTERY_SOC_BREAKPOINT_2) {",
        "        // Middle segment (linear)",
        "        return params->m * soc + params->b;",
        "    } ",
        "    else {",
        "        // Third segment (rational)",
        "        return (params->a3 + params->b3 * soc) / (params->c3 + params->d3 * soc);",
        "    }",
        "}",
        "",
        "// Helper function to calculate OCV slope for specific parameters and SOC",
        "static float calc_ocv_slope(const ocv_params_t* params, float soc) {",
        "    if (soc < BATTERY_SOC_BREAKPOINT_1) {",
        "        // First segment (rational)",
        "        float denominator = params->c1 + params->d1 * soc;",
        "        return (params->b1 * params->c1 - params->a1 * params->d1) / (denominator * denominator);",
        "    } ",
        "    else if (soc <= BATTERY_SOC_BREAKPOINT_2) {",
        "        // Middle segment (linear)",
        "        return params->m;",
        "    } ",
        "    else {",
        "        // Third segment (rational)",
        "        float denominator = params->c3 + params->d3 * soc;",
        "        return (params->b3 * params->c3 - params->a3 * params->d3) / (denominator * denominator);",
        "    }",
        "}",
        "",
        "// Helper function to calculate SOC from OCV for specific parameters",
        "static float calc_soc_from_ocv(const ocv_params_t* params, float ocv) {",
        "    // Calculate breakpoint voltages",
        "    float ocv_breakpoint_1 = calc_ocv(params, BATTERY_SOC_BREAKPOINT_1);",
        "    float ocv_breakpoint_2 = calc_ocv(params, BATTERY_SOC_BREAKPOINT_2);",
        "    ",
        "    if (ocv < ocv_breakpoint_1) {",
        "        // First segment (rational)",
        "        return (params->a1 - params->c1 * ocv) / (params->d1 * ocv - params->b1);",
        "    } ",
        "    else if (ocv <= ocv_breakpoint_2) {",
        "        // Middle segment (linear)",
        "        return (ocv - params->b) / params->m;",
        "    } ",
        "    else {",
        "        // Third segment (rational)",
        "        return (params->a3 - params->c3 * ocv) / (params->d3 * ocv - params->b3);",
        "    }",
        "}",
        "",
        "float battery_rint(float temperature) {",
        "    // Calculate R_int using rational function: (a + b*t)/(c + d*t)",
        "    float a = BATTERY_R_INT_PARAMS.a;",
        "    float b = BATTERY_R_INT_PARAMS.b;",
        "    float c = BATTERY_R_INT_PARAMS.c;",
        "    float d = BATTERY_R_INT_PARAMS.d;",
        "    ",
        "    return (a + b * temperature) / (c + d * temperature);",
        "}",
        "",
        "float battery_total_capacity(float temperature) {",
        "    // Handle out-of-bounds temperatures",
        "    if (temperature <= BATTERY_TEMP_POINTS[0]) {",
        "        return BATTERY_OCV_PARAMS[0].total_capacity;",
        "    }",
        "    ",
        "    if (temperature >= BATTERY_TEMP_POINTS[BATTERY_NUM_TEMPERATURE_POINTS - 1]) {",
        "        return BATTERY_OCV_PARAMS[BATTERY_NUM_TEMPERATURE_POINTS - 1].total_capacity;",
        "    }",
        "    ",
        "    // Find temperature bracket",
        "    for (int i = 0; i < BATTERY_NUM_TEMPERATURE_POINTS - 1; i++) {",
        "        if (temperature < BATTERY_TEMP_POINTS[i + 1]) {",
        "            return linear_interpolate(temperature,",
        "                                     BATTERY_TEMP_POINTS[i], BATTERY_OCV_PARAMS[i].total_capacity,",
        "                                     BATTERY_TEMP_POINTS[i + 1], BATTERY_OCV_PARAMS[i + 1].total_capacity);",
        "        }",
        "    }",
        "    ",
        "    // Should never reach here",
        "    return BATTERY_OCV_PARAMS[0].total_capacity;",
        "}",
        "",
        "float battery_meas_to_ocv(float voltage_V, float current_mA, float temperature) {",
        "    // Convert to mA to A by dividing by 1000",
        "    float current_A = current_mA / 1000.0f;",
        "    ",
        "    // Calculate OCV: V_OC = V_term + I * R_int",
        "    return voltage_V + (current_A * battery_rint(temperature));",
        "}",
        "",
        "float battery_ocv(float soc, float temperature) {",
        "    // Clamp SOC to valid range",
        "    soc = (soc < 0.0f) ? 0.0f : ((soc > 1.0f) ? 1.0f : soc);",
        "    ",
        "    // Handle out-of-bounds temperatures",
        "    if (temperature <= BATTERY_TEMP_POINTS[0]) {",
        "        return calc_ocv(&BATTERY_OCV_PARAMS[0], soc);",
        "    }",
        "    ",
        "    if (temperature >= BATTERY_TEMP_POINTS[BATTERY_NUM_TEMPERATURE_POINTS - 1]) {",
        "        return calc_ocv(&BATTERY_OCV_PARAMS[BATTERY_NUM_TEMPERATURE_POINTS - 1], soc);",
        "    }",
        "    ",
        "    // Find temperature bracket and interpolate",
        "    for (int i = 0; i < BATTERY_NUM_TEMPERATURE_POINTS - 1; i++) {",
        "        if (temperature < BATTERY_TEMP_POINTS[i + 1]) {",
        "            float ocv_low = calc_ocv(&BATTERY_OCV_PARAMS[i], soc);",
        "            float ocv_high = calc_ocv(&BATTERY_OCV_PARAMS[i + 1], soc);",
        "            ",
        "            return linear_interpolate(temperature,",
        "                                     BATTERY_TEMP_POINTS[i], ocv_low,",
        "                                     BATTERY_TEMP_POINTS[i + 1], ocv_high);",
        "        }",
        "    }",
        "    ",
        "    // Should never reach here",
        "    return calc_ocv(&BATTERY_OCV_PARAMS[0], soc);",
        "}",
        "",
        "float battery_ocv_slope(float soc, float temperature) {",
        "    // Clamp SOC to valid range",
        "    soc = (soc < 0.0f) ? 0.0f : ((soc > 1.0f) ? 1.0f : soc);",
        "    ",
        "    // Handle out-of-bounds temperatures",
        "    if (temperature <= BATTERY_TEMP_POINTS[0]) {",
        "        return calc_ocv_slope(&BATTERY_OCV_PARAMS[0], soc);",
        "    }",
        "    ",
        "    if (temperature >= BATTERY_TEMP_POINTS[BATTERY_NUM_TEMPERATURE_POINTS - 1]) {",
        "        return calc_ocv_slope(&BATTERY_OCV_PARAMS[BATTERY_NUM_TEMPERATURE_POINTS - 1], soc);",
        "    }",
        "    ",
        "    // Find temperature bracket and interpolate",
        "    for (int i = 0; i < BATTERY_NUM_TEMPERATURE_POINTS - 1; i++) {",
        "        if (temperature < BATTERY_TEMP_POINTS[i + 1]) {",
        "            float slope_low = calc_ocv_slope(&BATTERY_OCV_PARAMS[i], soc);",
        "            float slope_high = calc_ocv_slope(&BATTERY_OCV_PARAMS[i + 1], soc);",
        "            ",
        "            return linear_interpolate(temperature,",
        "                                     BATTERY_TEMP_POINTS[i], slope_low,",
        "                                     BATTERY_TEMP_POINTS[i + 1], slope_high);",
        "        }",
        "    }",
        "    ",
        "    // Should never reach here",
        "    return calc_ocv_slope(&BATTERY_OCV_PARAMS[0], soc);",
        "}",
        "",
        "float battery_soc(float ocv, float temperature) {",
        "    // Handle out-of-bounds temperatures",
        "    if (temperature <= BATTERY_TEMP_POINTS[0]) {",
        "        return calc_soc_from_ocv(&BATTERY_OCV_PARAMS[0], ocv);",
        "    }",
        "    ",
        "    if (temperature >= BATTERY_TEMP_POINTS[BATTERY_NUM_TEMPERATURE_POINTS - 1]) {",
        "        return calc_soc_from_ocv(&BATTERY_OCV_PARAMS[BATTERY_NUM_TEMPERATURE_POINTS - 1], ocv);",
        "    }",
        "    ",
        "    // Find temperature bracket and interpolate",
        "    for (int i = 0; i < BATTERY_NUM_TEMPERATURE_POINTS - 1; i++) {",
        "        if (temperature < BATTERY_TEMP_POINTS[i + 1]) {",
        "            float soc_low = calc_soc_from_ocv(&BATTERY_OCV_PARAMS[i], ocv);",
        "            float soc_high = calc_soc_from_ocv(&BATTERY_OCV_PARAMS[i + 1], ocv);",
        "            ",
        "            return linear_interpolate(temperature,",
        "                                     BATTERY_TEMP_POINTS[i], soc_low,",
        "                                     BATTERY_TEMP_POINTS[i + 1], soc_high);",
        "        }",
        "    }",
        "    ",
        "    // Should never reach here",
        "    return calc_soc_from_ocv(&BATTERY_OCV_PARAMS[0], ocv);",
        "}"
    ]

    # Write the implementation to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(impl))

    print(f"Battery model implementation file generated: {output_path}")


def generate_ekf_header(output_path="battery_ekf.h"):
    """
    Generate C header file for the EKF fuel gauge functions

    Parameters:
    - output_path: Path to save the header file
    """
    header = [
        "/**",
        " * Battery Extended Kalman Filter (EKF) Fuel Gauge",
        " * Auto-generated from EkfEstimator Python class",
        " */",
        "",
        "#ifndef BATTERY_EKF_H",
        "#define BATTERY_EKF_H",
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
        "#endif // BATTERY_EKF_H"
    ]

    # Write the header to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(header))

    print(f"EKF header file generated: {output_path}")


def generate_ekf_implementation(output_path="battery_ekf.c"):
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
        "    // Calculate OCV from terminal voltage and current",
        "    float ocv = battery_meas_to_ocv(voltage_V, current_mA, temperature);",
        "    ",
        "    // Get SOC from OCV using lookup",
        "    state->soc = battery_soc(ocv, temperature);",
        "    state->soc_latched = state->soc;",
        "}",
        "",
        "float battery_ekf_update(battery_ekf_state_t* state, uint32_t dt,",
        "                         float voltage_V, float current_mA, float temperature) {",
        "    // Filter measurements",
        "    voltage_V = filter_measurement(state, voltage_V, state->v_history);",
        "    current_mA = filter_measurement(state, current_mA, state->i_history);",
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
        "    float total_capacity = battery_total_capacity(temperature);",
        "    ",
        "    // State prediction (coulomb counting)",
        "    // SOC_k+1 = SOC_k - (I*dt)/(3600*capacity)",
        "    float x_k1_k = state->soc - (current_mA / (3600.0f * total_capacity)) * dt_sec;",
        "    ",
        "    // Calculate Jacobian of measurement function h(x) = dOCV/dSOC",
        "    float h_jacobian = battery_ocv_slope(x_k1_k, temperature);",
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
        "    float v_pred = battery_ocv(x_k1_k, temperature) - (current_mA / 1000.0f) * battery_rint(temperature);",
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


def export_battery_model_lookup(battery_model_data, output_dir="."):
    """
    Export the battery model data to C code files

    Parameters:
    - battery_model_data: Dictionary containing 'r_int' parameters and 'ocv_curves' data
    - output_dir: Directory to save the output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate battery model files
    battery_model_h = os.path.join(output_dir, "battery_model.h")
    battery_model_c = os.path.join(output_dir, "battery_model.c")

    generate_battery_model_header(battery_model_data, battery_model_h)
    generate_battery_model_implementation(battery_model_data, battery_model_c)

    # Generate EKF fuel gauge files
    ekf_h = os.path.join(output_dir, "battery_ekf.h")
    ekf_c = os.path.join(output_dir, "battery_ekf.c")

    generate_ekf_header(ekf_h)
    generate_ekf_implementation(ekf_c)

    print(f"Battery model and EKF fuel gauge files exported to {output_dir}")

    # For debugging - save the battery model data to JSON
    # json_path = os.path.join(output_dir, "battery_model_data.json")
    # with open(json_path, 'w') as f:
    #     json.dump({str(k): v for k, v in battery_model_data.items()}, f, indent=2)


if __name__ == "__main__":
    # Example usage
    from run_battery_characterization import main as run_characterization

    # If this script is run directly, try to import the battery model data
    # from a parent script that has already run the characterization
    try:
        # Import battery_model_data from main script if available
        from __main__ import battery_model_data
        export_battery_model_lookup(battery_model_data, "generated_output")
    except (ImportError, AttributeError):
        print("No battery_model_data found in parent script.")
        print("Please run this script after battery characterization or provide the data explicitly.")