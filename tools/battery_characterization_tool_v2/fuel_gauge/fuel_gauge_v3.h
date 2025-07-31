/**
 * Adaptive EKF Fuel Gauge Algorithm for LiFePO4 Batteries
 *
 * C implementation of adaptive battery state of charge estimation algorithm
 * using Extended Kalman filtering with temperature-dependent battery models.
 */

#ifndef FUEL_GAUGE_H
#define FUEL_GAUGE_H

#include <stdint.h>
#include <stdbool.h>

// Maximum number of samples for measurement filtering
#define FILTER_WINDOW_SIZE 5

// Fuel gauge structure to track state
typedef struct {
    float state_of_charge;       // SoC estimate (x in Python)
    float latched_soc;           // Latched SoC value for adaptive algorithm
    float error_covariance;      // Estimation error covariance (P in Python)
    float process_noise;         // Process noise covariance (Q in Python)
    float measurement_noise;     // Measurement noise covariance (R in Python)

    // Default/configured parameters
    float process_noise_default;
    float measurement_noise_default;
    float process_noise_aggressive;  // Higher process noise for low SoC
    float measurement_noise_aggressive; // Higher measurement noise for low SoC
    float initial_covariance;    // Initial error covariance

    // Measurement filtering
    float voltage_history[FILTER_WINDOW_SIZE];
    float current_history[FILTER_WINDOW_SIZE];
    uint8_t history_count;
} fuel_gauge_t;

/**
 * Initialize the fuel gauge with Kalman filter parameters
 *
 * @param fg Pointer to fuel gauge structure
 * @param Q Process noise covariance (default)
 * @param R Measurement noise covariance (default)
 * @param Q_aggressive Higher process noise for low SoC regions
 * @param R_aggressive Higher measurement noise for low SoC regions
 * @param P_init Initial error covariance
 */
void fuel_gauge_init(fuel_gauge_t *fg, float Q, float R, float Q_aggressive,
                    float R_aggressive, float P_init);

/**
 * Reset the fuel gauge state
 *
 * @param fg Pointer to fuel gauge structure
 */
void fuel_gauge_reset(fuel_gauge_t *fg);

/**
 * Use the first measurement to initialize the state of charge
 *
 * @param fg Pointer to fuel gauge structure
 * @param V_meas Measured battery voltage (V)
 * @param I_meas Measured battery current (mA, positive = discharge)
 * @param T_meas Measured battery temperature (°C)
 * @param override_init_soc Optional initial SOC override (0.0-1.0, or -1 for auto)
 */
void fuel_gauge_initial_guess(fuel_gauge_t *fg, float V_meas, float I_meas,
                             float T_meas, float override_init_soc);

/**
 * Simple guess of SoC based on measurements (without Kalman filter)
 *
 * @param V_meas Measured battery voltage (V)
 * @param I_meas Measured battery current (mA, positive = discharge)
 * @param T_meas Measured battery temperature (°C)
 * @return Estimated state of charge (0.0 to 1.0)
 */
float fuel_gauge_simple_guess(float V_meas, float I_meas, float T_meas);

/**
 * Update the fuel gauge state using adaptive extended Kalman filter
 * This version adapts filter parameters based on SoC level and
 * implements SoC latching based on current direction
 *
 * @param fg Pointer to fuel gauge structure
 * @param dt Time since last update (milliseconds)
 * @param V_meas Measured battery voltage (V)
 * @param I_meas Measured battery current (mA, positive = discharge)
 * @param T_meas Measured battery temperature (°C)
 * @return Updated latched state of charge estimate (0.0 to 1.0)
 */
float fuel_gauge_update_ekf_adaptive(fuel_gauge_t *fg, float dt, float V_meas,
                                    float I_meas, float T_meas);

/**
 * Get the current SoC estimate
 *
 * @param fg Pointer to fuel gauge structure
 * @return Current state of charge estimate (0.0 to 1.0)
 */
float fuel_gauge_get_soc(const fuel_gauge_t *fg);

/**
 * Get the latched SoC estimate (from adaptive filter)
 *
 * @param fg Pointer to fuel gauge structure
 * @return Latched state of charge estimate (0.0 to 1.0)
 */
float fuel_gauge_get_latched_soc(const fuel_gauge_t *fg);

/**
 * Convert measured battery voltage to open-circuit voltage
 * by compensating for IR drop
 *
 * @param V_meas Measured battery voltage (V)
 * @param I_meas Measured battery current (mA, positive = discharge)
 * @param T_meas Measured battery temperature (°C)
 * @return Calculated open circuit voltage (V)
 */
float fuel_gauge_meas_to_ocv(float V_meas, float I_meas, float T_meas);

#endif // FUEL_GAUGE_H