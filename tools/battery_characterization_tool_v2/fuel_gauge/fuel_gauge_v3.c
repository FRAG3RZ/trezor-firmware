/**
 * Adaptive EKF Fuel Gauge Algorithm Implementation
 */

#include "fuel_gauge.h"
#include "battery_lookup_tables.h"
#include <math.h>

// Helper function to filter measurements
static float filter_measurement(float new_value, float *history, uint8_t *count, uint8_t max_count) {
    // Shift values if buffer is full
    if (*count >= max_count) {
        for (uint8_t i = 0; i < max_count - 1; i++) {
            history[i] = history[i + 1];
        }
        history[max_count - 1] = new_value;
    } else {
        // Add new value to history
        history[*count] = new_value;
        (*count)++;
    }

    // Calculate average
    float sum = 0.0f;
    for (uint8_t i = 0; i < *count; i++) {
        sum += history[i];
    }
    return sum / (*count);
}

// Helper function for linear interpolation
static float linear_interpolate(float x, float x1, float y1, float x2, float y2) {
    // Check for division by zero (or very small denominator)
    if (fabsf(x2 - x1) < 1e-6f) {
        return (y1 + y2) / 2.0f;  // Return average if x values are too close
    }
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1);
}

// Initialize the fuel gauge
void fuel_gauge_init(fuel_gauge_t *fg, float Q, float R, float Q_aggressive,
                    float R_aggressive, float P_init) {
    fg->state_of_charge = 0.0f;
    fg->latched_soc = 0.0f;
    fg->error_covariance = P_init;
    fg->process_noise = Q;
    fg->measurement_noise = R;
    fg->process_noise_default = Q;
    fg->measurement_noise_default = R;
    fg->process_noise_aggressive = Q_aggressive;
    fg->measurement_noise_aggressive = R_aggressive;
    fg->initial_covariance = P_init;
    fg->history_count = 0;
}

// Reset the fuel gauge state
void fuel_gauge_reset(fuel_gauge_t *fg) {
    fg->state_of_charge = 0.0f;
    fg->latched_soc = 0.0f;
    fg->error_covariance = fg->initial_covariance;
    fg->process_noise = fg->process_noise_default;
    fg->measurement_noise = fg->measurement_noise_default;
    fg->history_count = 0;
}

// Convert measured battery voltage to open-circuit voltage
float fuel_gauge_meas_to_ocv(float V_meas, float I_meas, float T_meas) {
    return V_meas + ((I_meas / 1000.0f) * battery_get_internal_resistance(T_meas));
}

// Simple guess of SoC based on measurements (without Kalman filter)
float fuel_gauge_simple_guess(float V_meas, float I_meas, float T_meas) {
    float ocv = fuel_gauge_meas_to_ocv(V_meas, I_meas, T_meas);
    return battery_get_soc(ocv, T_meas);
}

// Use the first measurement to initialize the state of charge
void fuel_gauge_initial_guess(fuel_gauge_t *fg, float V_meas, float I_meas,
                             float T_meas, float override_init_soc) {
    float ocv = fuel_gauge_meas_to_ocv(V_meas, I_meas, T_meas);
    fg->state_of_charge = battery_get_soc(ocv, T_meas);
    fg->latched_soc = fg->state_of_charge;

    // Override initial SOC if requested
    if (override_init_soc >= 0.0f && override_init_soc <= 1.0f) {
        fg->state_of_charge = override_init_soc;
        fg->latched_soc = override_init_soc;
    }

    fg->error_covariance = fg->initial_covariance;
}

// Calculate the slope of the OCV-SOC curve for the Jacobian in EKF
static float calculate_ocv_slope(float soc, float temperature) {
    const float delta = 0.01f;  // Small delta for numerical differentiation

    // Make sure we don't go out of bounds
    float soc_plus = (soc + delta > 1.0f) ? 1.0f : (soc + delta);
    float soc_minus = (soc - delta < 0.0f) ? 0.0f : (soc - delta);

    // Calculate OCVs at nearby points
    float voc_plus = battery_get_voc(soc_plus, temperature);
    float voc_minus = battery_get_voc(soc_minus, temperature);

    // Calculate slope via central difference
    return (voc_plus - voc_minus) / (soc_plus - soc_minus);
}

// Adaptive Extended Kalman Filter implementation
float fuel_gauge_update_ekf_adaptive(fuel_gauge_t *fg, float dt, float V_meas, float I_meas, float T_meas) {
    // Filter measurements
    V_meas = filter_measurement(V_meas, fg->voltage_history, &fg->history_count, FILTER_WINDOW_SIZE);
    I_meas = filter_measurement(I_meas, fg->current_history, &fg->history_count, FILTER_WINDOW_SIZE);

    // Adjust parameters based on temperature
    if (T_meas < 10.0f) {
        fg->measurement_noise = 10.0f;  // Higher measurement noise at low temps
        fg->process_noise = 0.01f;      // Higher process noise at low temps
    } else {
        // Adapt parameters based on SOC level
        if (fg->latched_soc < 0.2f) {
            // More aggressive filter settings for low SOC
            fg->process_noise = fg->process_noise_aggressive;
            fg->measurement_noise = fg->measurement_noise_aggressive;
        } else {
            // Default filter settings for normal SOC range
            fg->process_noise = fg->process_noise_default;
            fg->measurement_noise = fg->measurement_noise_default;
        }
    }

    // Convert time to seconds
    float dt_seconds = dt / 1000.0f;

    // Get battery capacity at current temperature
    float total_capacity = battery_get_capacity(T_meas);

    // State prediction (coulomb counting)
    float x_k1_k = fg->state_of_charge - (I_meas / (3600.0f * total_capacity)) * dt_seconds;

    // Calculate Jacobian of measurement function h(x)
    float h_jacobian = calculate_ocv_slope(x_k1_k, T_meas);

    // Hardcoded value is used in the Python code - can be 1.0 as a fallback
    if (fabsf(h_jacobian) < 1e-6f) {
        h_jacobian = 1.0f;
    }

    // Error covariance prediction
    float P_k1_k = fg->error_covariance + fg->process_noise;

    // Calculate innovation covariance
    float S = h_jacobian * P_k1_k * h_jacobian + fg->measurement_noise;

    // Prevent division by zero
    if (fabsf(S) < 1e-6f) {
        S = 1e-6f;
    }

    // Calculate Kalman gain
    float K_k1_k = P_k1_k * h_jacobian / S;

    // Calculate predicted terminal voltage
    float r_int = battery_get_internal_resistance(T_meas);
    float v_pred = battery_get_voc(x_k1_k, T_meas) - (I_meas / 1000.0f) * r_int;

    // State update
    float x_k1_k1 = x_k1_k + K_k1_k * (V_meas - v_pred);

    // Error covariance update
    float P_k1_k1 = (1.0f - K_k1_k * h_jacobian) * P_k1_k;

    // Enforce SoC boundaries
    fg->state_of_charge = (x_k1_k1 < 0.0f) ? 0.0f : ((x_k1_k1 > 1.0f) ? 1.0f : x_k1_k1);
    fg->error_covariance = P_k1_k1;

    // Based on current direction, update the latched SoC
    if (I_meas > 0.0f) {
        // Discharging - SoC should only decrease
        if (fg->state_of_charge < fg->latched_soc) {
            fg->latched_soc = fg->state_of_charge;
        }
    } else {
        // Charging - SoC should only increase
        if (fg->state_of_charge > fg->latched_soc) {
            fg->latched_soc = fg->state_of_charge;
        }
    }

    return fg->latched_soc;
}

// Get the current SoC estimate
float fuel_gauge_get_soc(const fuel_gauge_t *fg) {
    return fg->state_of_charge;
}

// Get the latched SoC estimate (from adaptive filter)
float fuel_gauge_get_latched_soc(const fuel_gauge_t *fg) {
    return fg->latched_soc;
}