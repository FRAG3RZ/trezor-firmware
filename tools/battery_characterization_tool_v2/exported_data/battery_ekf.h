/**
 * Battery Extended Kalman Filter (EKF) Fuel Gauge
 * Auto-generated from EkfEstimator Python class
 */

#ifndef BATTERY_EKF_H
#define BATTERY_EKF_H

#include <stdint.h>
#include <stdbool.h>

// EKF State structure
typedef struct {
    // State estimate (SOC)
    float soc;
    // Latched SOC (the one that gets reported)
    float soc_latched;
    // Error covariance
    float P;
    // Filter parameters
    float R;         // Measurement noise variance
    float Q;         // Process noise variance
    float R_aggressive;  // Aggressive measurement noise variance
    float Q_aggressive;  // Aggressive process noise variance
    // Measurement history for filtering
    float v_history[5];  // Voltage history buffer
    float i_history[5];  // Current history buffer
    uint8_t history_count; // Number of samples in history
    uint8_t history_index; // Current index in circular buffer
} battery_ekf_state_t;

/**
 * Initialize the EKF state
 * @param state Pointer to EKF state structure
 * @param R Measurement noise variance
 * @param Q Process noise variance
 * @param R_aggressive Aggressive mode measurement noise variance
 * @param Q_aggressive Aggressive mode process noise variance
 * @param P_init Initial error covariance
 */
void battery_ekf_init(battery_ekf_state_t* state, float R, float Q,
                      float R_aggressive, float Q_aggressive, float P_init);

/**
 * Reset the EKF state
 * @param state Pointer to EKF state structure
 */
void battery_ekf_reset(battery_ekf_state_t* state);

/**
 * Make initial SOC guess based on OCV
 * @param state Pointer to EKF state structure
 * @param voltage_V Current battery voltage (V)
 * @param current_mA Current battery current (mA), positive for discharge
 * @param temperature Battery temperature (°C)
 */
void battery_ekf_initial_guess(battery_ekf_state_t* state,
                               float voltage_V, float current_mA, float temperature);

/**
 * Update the EKF with new measurements
 * @param state Pointer to EKF state structure
 * @param dt Time step in milliseconds
 * @param voltage_V Current battery voltage (V)
 * @param current_mA Current battery current (mA), positive for discharge
 * @param temperature Battery temperature (°C)
 * @return Updated SOC estimate (0.0 to 1.0)
 */
float battery_ekf_update(battery_ekf_state_t* state, uint32_t dt,
                         float voltage_V, float current_mA, float temperature);

#endif // BATTERY_EKF_H