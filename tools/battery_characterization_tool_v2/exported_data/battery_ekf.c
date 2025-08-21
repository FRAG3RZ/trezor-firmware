/**
 * Battery Extended Kalman Filter (EKF) Fuel Gauge Implementation
 * Auto-generated from EkfEstimator Python class
 */

#include "battery_ekf.h"
#include "battery_model.h"
#include <math.h>

// Filter window size for measurements
#define FILTER_WINDOW_SIZE 5

// Helper function for filtered measurements
static float filter_measurement(battery_ekf_state_t* state, float new_value, float* history) {
    // Add new measurement to history
    history[state->history_index] = new_value;
    
    // Update index for circular buffer
    state->history_index = (state->history_index + 1) % FILTER_WINDOW_SIZE;
    
    // Update count for averaging
    if (state->history_count < FILTER_WINDOW_SIZE) {
        state->history_count++;
    }
    
    // Calculate average
    float sum = 0.0f;
    for (uint8_t i = 0; i < state->history_count; i++) {
        sum += history[i];
    }
    
    return sum / state->history_count;
}

void battery_ekf_init(battery_ekf_state_t* state, float R, float Q,
                      float R_aggressive, float Q_aggressive, float P_init) {
    state->R = R;
    state->Q = Q;
    state->R_aggressive = R_aggressive;
    state->Q_aggressive = Q_aggressive;
    state->P = P_init;
    
    // Initialize state
    state->soc = 0.0f;
    state->soc_latched = 0.0f;
    state->history_count = 0;
    state->history_index = 0;
    
    // Initialize history arrays
    for (int i = 0; i < FILTER_WINDOW_SIZE; i++) {
        state->v_history[i] = 0.0f;
        state->i_history[i] = 0.0f;
    }
}

void battery_ekf_reset(battery_ekf_state_t* state) {
    // Reset state but keep filter parameters
    state->soc = 0.0f;
    state->soc_latched = 0.0f;
    state->history_count = 0;
    state->history_index = 0;
    
    // Reset history arrays
    for (int i = 0; i < FILTER_WINDOW_SIZE; i++) {
        state->v_history[i] = 0.0f;
        state->i_history[i] = 0.0f;
    }
}

void battery_ekf_initial_guess(battery_ekf_state_t* state,
                               float voltage_V, float current_mA, float temperature) {
    // Determine if we're in discharge mode
    bool discharging_mode = current_mA >= 0.0f;

    // Calculate OCV from terminal voltage and current
    float ocv = battery_meas_to_ocv(voltage_V, current_mA, temperature);
    
    // Get SOC from OCV using lookup
    state->soc = battery_soc(ocv, temperature, discharging_mode);
    state->soc_latched = state->soc;
}

float battery_ekf_update(battery_ekf_state_t* state, uint32_t dt,
                         float voltage_V, float current_mA, float temperature) {
    // Filter measurements
    voltage_V = filter_measurement(state, voltage_V, state->v_history);
    current_mA = filter_measurement(state, current_mA, state->i_history);
    
    // Determine if we're in discharge mode
    bool discharging_mode = current_mA >= 0.0f;
    
    // Choose filter parameters based on temperature and SOC
    float R = state->R;
    float Q = state->Q;
    
    if (temperature < 10.0f) {
        // Cold temperature - use more conservative values
        R = 10.0f;
        Q = 0.01f;
    } else if (state->soc_latched < 0.2f) {
        // Low SOC - use aggressive values to track more closely
        R = state->R_aggressive;
        Q = state->Q_aggressive;
    }
    
    // Convert milliseconds to seconds
    float dt_sec = dt / 1000.0f;
    
    // Get total capacity at current temperature
    float total_capacity = battery_total_capacity(temperature, discharging_mode);
    
    // State prediction (coulomb counting)
    // SOC_k+1 = SOC_k - (I*dt)/(3600*capacity)
    float x_k1_k = state->soc - (current_mA / (3600.0f * total_capacity)) * dt_sec;
    
    // Calculate Jacobian of measurement function h(x) = dOCV/dSOC
    float h_jacobian = battery_ocv_slope(x_k1_k, temperature, discharging_mode);
    
    // Error covariance prediction
    float P_k1_k = state->P + Q;
    
    // Calculate innovation covariance
    float S = h_jacobian * P_k1_k * h_jacobian + R;
    
    // Calculate Kalman gain
    float K_k1_k = P_k1_k * h_jacobian / S;
    
    // Calculate predicted terminal voltage
    float v_pred = battery_ocv(x_k1_k, temperature, discharging_mode) - 
                   (current_mA / 1000.0f) * battery_rint(temperature);
    
    // State update
    float x_k1_k1 = x_k1_k + K_k1_k * (voltage_V - v_pred);
    
    // Error covariance update
    float P_k1_k1 = (1.0f - K_k1_k * h_jacobian) * P_k1_k;
    
    // Enforce SOC boundaries
    state->soc = (x_k1_k1 < 0.0f) ? 0.0f : ((x_k1_k1 > 1.0f) ? 1.0f : x_k1_k1);
    state->P = P_k1_k1;
    
    // Update latched SOC based on current direction
    if (current_mA > 0.0f) {
        // Discharging, SOC should move only in negative direction
        if (state->soc < state->soc_latched) {
            state->soc_latched = state->soc;
        }
    } else {
        // Charging, SOC should move only in positive direction
        if (state->soc > state->soc_latched) {
            state->soc_latched = state->soc;
        }
    }
    
    return state->soc_latched;
}