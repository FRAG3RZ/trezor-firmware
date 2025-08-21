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

// Temperature points arrays (in Celsius)
// Discharge temperatures
static const float BATTERY_TEMP_POINTS_DISCHG[BATTERY_NUM_TEMP_POINTS] = {
    15.55f,     20.54f,     25.46f,     31.41f
};

// Charge temperatures
static const float BATTERY_TEMP_POINTS_CHG[BATTERY_NUM_TEMP_POINTS] = {
    17.51f,     22.46f,     27.54f,     32.31f
};

// Internal resistance curve parameters (rational function parameters a+b*t)/(c+d*t)
static const float BATTERY_R_INT_PARAMS[4] = {
    // a, b, c, d for rational function (a + b*t)/(c + d*t)
    10.226749f, 0.634799f, 4.511901f, 2.038260f
};

// Discharge OCV curve parameters for each temperature
static const float BATTERY_OCV_DISCHARGE_PARAMS[BATTERY_NUM_TEMP_POINTS][10] = {
    // Temperature: 15.55°C (key: 15)
    {
        0.114206f, 3.232050f, // m, b (linear segment)
        14.892594f, 138.112728f, 4.859967f, 41.187931f, // a1, b1, c1, d1 (first rational segment)
        1200.027376f, -1196.626018f, 361.207214f, -360.214908f  // a3, b3, c3, d3 (third rational segment)
    },
    // Temperature: 20.54°C (key: 20)
    {
        0.114947f, 3.233992f, // m, b (linear segment)
        424.471372f, 4847.344144f, 138.412901f, 1452.408188f, // a1, b1, c1, d1 (first rational segment)
        13568.203607f, -13529.721549f, 4080.982206f, -4069.783172f  // a3, b3, c3, d3 (third rational segment)
    },
    // Temperature: 25.46°C (key: 25)
    {
        0.117254f, 3.234777f, // m, b (linear segment)
        54.277079f, 786.933714f, 17.729756f, 236.683391f, // a1, b1, c1, d1 (first rational segment)
        2557.887452f, -2549.463878f, 768.799815f, -766.353566f  // a3, b3, c3, d3 (third rational segment)
    },
    // Temperature: 31.41°C (key: 30)
    {
        0.117392f, 3.238201f, // m, b (linear segment)
        121.263204f, 1959.216479f, 39.641450f, 589.477829f, // a1, b1, c1, d1 (first rational segment)
        1125.876897f, -1123.811778f, 337.973980f, -337.376068f  // a3, b3, c3, d3 (third rational segment)
    }
};

// Charge OCV curve parameters for each temperature
static const float BATTERY_OCV_CHARGE_PARAMS[BATTERY_NUM_TEMP_POINTS][10] = {
    // Temperature: 17.51°C (key: 15)
    {
        0.115636f, 3.276418f, // m, b (linear segment)
        35.139163f, 290.354744f, 10.994941f, 86.389017f, // a1, b1, c1, d1 (first rational segment)
        1188.566970f, -1146.975807f, 356.593735f, -345.195918f  // a3, b3, c3, d3 (third rational segment)
    },
    // Temperature: 22.46°C (key: 20)
    {
        0.118045f, 3.276057f, // m, b (linear segment)
        -64.437642f, -488.479720f, -20.239748f, -144.790482f, // a1, b1, c1, d1 (first rational segment)
        50.222216f, -48.430984f, 15.063535f, -14.572835f  // a3, b3, c3, d3 (third rational segment)
    },
    // Temperature: 27.54°C (key: 25)
    {
        0.111074f, 3.280859f, // m, b (linear segment)
        29.065740f, 389.157685f, 9.125533f, 116.256271f, // a1, b1, c1, d1 (first rational segment)
        1116.902277f, -1081.560350f, 334.676432f, -324.994265f  // a3, b3, c3, d3 (third rational segment)
    },
    // Temperature: 32.31°C (key: 30)
    {
        0.105865f, 3.277899f, // m, b (linear segment)
        70.614775f, 661.648369f, 22.273192f, 196.622496f, // a1, b1, c1, d1 (first rational segment)
        1055.899642f, -1040.952709f, 315.380880f, -311.277599f  // a3, b3, c3, d3 (third rational segment)
    }
};

// Battery capacity data for each temperature
static const float BATTERY_CAPACITY[BATTERY_NUM_TEMP_POINTS][2] = {
    // Temperature: 15.55°C (key: 15)
    { 355.86f, 378.79f },
    // Temperature: 20.54°C (key: 20)
    { 365.45f, 392.83f },
    // Temperature: 25.46°C (key: 25)
    { 369.62f, 375.32f },
    // Temperature: 31.41°C (key: 30)
    { 361.17f, 379.75f }
};

#endif // BATTERY_DATA_JYHPFL333838_H