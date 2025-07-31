/**
 * Battery Data: JYHPFL333838
 * Auto-generated from battery characterization data
 * Contains lookup tables and parameters for the specific battery model
 */

#ifndef BATTERY_DATA_JYHPFL333838_H
#define BATTERY_DATA_JYHPFL333838_H

#include <stdint.h>

/**
 * Battery Specifications:
 * Model: JYHPFL333838
 * Chemistry: LiFePO4
 * Characterized on: TODO - Add date
 */

// Configuration
#define BATTERY_NUM_TEMP_POINTS 4

// SOC breakpoints for piecewise functions
#define BATTERY_SOC_BREAKPOINT_1 0.25f
#define BATTERY_SOC_BREAKPOINT_2 0.8f

// Temperature points array (in Celsius)
static const float BATTERY_TEMP_POINTS[BATTERY_NUM_TEMP_POINTS] = {
    16.22f, 21.07f, 26.08f, 31.01f
};

// Internal resistance curve parameters (rational function parameters a+b*t)/(c+d*t)
static const float BATTERY_R_INT_PARAMS[4] = {
    // a, b, c, d for rational function (a + b*t)/(c + d*t)
    0.345854f, 0.000708f, 0.331842f, 0.016293f
};

// Discharge OCV curve parameters for each temperature
static const float BATTERY_OCV_DISCHARGE_PARAMS[BATTERY_NUM_TEMP_POINTS][10] = {
    // Temperature: 16.22°C
    {
        0.104960f, 3.241228f, // m, b (linear segment)
        0.352956f, 1.953420f, 0.113863f, 0.576831f, // a1, b1, c1, d1 (first rational segment)
        1.451425f, -1.446630f, 0.436666f, -0.435267f  // a3, b3, c3, d3 (third rational segment)
    },
    // Temperature: 21.07°C
    {
        0.107647f, 3.243069f, // m, b (linear segment)
        0.244383f, 1.960926f, 0.079021f, 0.583995f, // a1, b1, c1, d1 (first rational segment)
        1.647271f, -1.641452f, 0.494989f, -0.493288f  // a3, b3, c3, d3 (third rational segment)
    },
    // Temperature: 26.08°C
    {
        0.110229f, 3.237182f, // m, b (linear segment)
        0.181800f, 1.983370f, 0.059086f, 0.594844f, // a1, b1, c1, d1 (first rational segment)
        1.088183f, -1.083547f, 0.327401f, -0.326047f  // a3, b3, c3, d3 (third rational segment)
    },
    // Temperature: 31.01°C
    {
        0.112294f, 3.242170f, // m, b (linear segment)
        0.170607f, 1.985012f, 0.055424f, 0.594804f, // a1, b1, c1, d1 (first rational segment)
        1.600459f, -1.597085f, 0.480466f, -0.479487f  // a3, b3, c3, d3 (third rational segment)
    }
};

// Charge OCV curve parameters for each temperature
static const float BATTERY_OCV_CHARGE_PARAMS[BATTERY_NUM_TEMP_POINTS][10] = {
    // Temperature: 16.22°C
    {
        0.119823f, 3.262298f, // m, b (linear segment)
        0.261473f, 2.112202f, 0.082254f, 0.630477f, // a1, b1, c1, d1 (first rational segment)
        1.480603f, -1.421921f, 0.446259f, -0.430125f  // a3, b3, c3, d3 (third rational segment)
    },
    // Temperature: 21.07°C
    {
        0.124349f, 3.260526f, // m, b (linear segment)
        0.181307f, 1.879065f, 0.057377f, 0.561790f, // a1, b1, c1, d1 (first rational segment)
        0.953134f, -0.902039f, 0.288210f, -0.274139f  // a3, b3, c3, d3 (third rational segment)
    },
    // Temperature: 26.08°C
    {
        0.106731f, 3.291650f, // m, b (linear segment)
        0.181070f, 2.929839f, 0.056227f, 0.875681f, // a1, b1, c1, d1 (first rational segment)
        0.219883f, -0.207869f, 0.066117f, -0.062811f  // a3, b3, c3, d3 (third rational segment)
    },
    // Temperature: 31.01°C
    {
        0.102116f, 3.277235f, // m, b (linear segment)
        0.333816f, 2.714662f, 0.104485f, 0.808770f, // a1, b1, c1, d1 (first rational segment)
        0.536240f, -0.527753f, 0.160367f, -0.158020f  // a3, b3, c3, d3 (third rational segment)
    }
};

// Battery capacity data for each temperature
static const float BATTERY_CAPACITY[BATTERY_NUM_TEMP_POINTS][2] = {
    // Temperature: 16.22°C
    { 0.36f, 0.39f },
    // Temperature: 21.07°C
    { 0.37f, 0.40f },
    // Temperature: 26.08°C
    { 0.37f, 0.36f },
    // Temperature: 31.01°C
    { 0.36f, 0.37f }
};

#endif // BATTERY_DATA_JYHPFL333838_H