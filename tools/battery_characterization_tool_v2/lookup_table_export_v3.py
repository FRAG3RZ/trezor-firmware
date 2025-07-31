def generate_battery_lookup_header(soc_curves, output_path="battery_lookup_tables.h"):
    """
    Generate C header file from Python soc_curves dictionary

    Parameters:
    - soc_curves: Dictionary of {temp: {'curve': [soc_array, voc_array], 'r_int': value, 'total_capacity': value}}
    - output_path: Path to save the header file
    """
    # Get key info from the dictionary
    temp_points = sorted(list(soc_curves.keys()))
    num_temp_points = len(temp_points)

    # Get the number of SoC points from the first temperature entry
    first_temp = temp_points[0]
    num_soc_points = len(soc_curves[first_temp]['curve'][0])

    # Determine if voltage increases or decreases with SOC (for LiFePO4, voltage typically decreases with SOC)
    voc_array = soc_curves[first_temp]['curve'][1]
    voltage_decreases = voc_array[0] > voc_array[-1]

    # Start building the header file content
    header = [
        "/**",
        " * Battery SoC Lookup Tables",
        " * Auto-generated from Python battery characterization script",
        " * Battery type: LiFePO4",
        " */",
        "",
        "#ifndef BATTERY_LOOKUP_TABLES_H",
        "#define BATTERY_LOOKUP_TABLES_H",
        "",
        "#include <stdint.h>",
        "#include <stdbool.h>",
        "",
        "// Configuration",
        f"#define NUM_TEMPERATURE_POINTS {num_temp_points}",
        f"#define NUM_SOC_POINTS {num_soc_points}",
        f"#define VOC_DECREASES_WITH_SOC {1 if voltage_decreases else 0}  // Indicates if voltage decreases as SOC decreases",
        "",
        "// Temperature points array (in Celsius)",
        "static const float BATTERY_TEMP_POINTS[NUM_TEMPERATURE_POINTS] = {",
        "    " + ", ".join([f"{temp:.2f}f" for temp in temp_points]),
        "};",
        "",
        "// Internal resistance array (in ohms) corresponding to each temperature",
        "static const float BATTERY_R_INT[NUM_TEMPERATURE_POINTS] = {",
        "    " + ", ".join([f"{soc_curves[temp]['r_int']:.6f}f" for temp in temp_points]),
        "};",
        "",
        "// Total capacity array (in mAh) corresponding to each temperature",
        "static const float BATTERY_CAPACITY[NUM_TEMPERATURE_POINTS] = {",
        "    " + ", ".join([f"{soc_curves[temp]['total_capacity']:.2f}f" for temp in temp_points]),
        "};",
        "",
        "// SoC points (constant across all temperature curves)",
        "static const float BATTERY_SOC_POINTS[NUM_SOC_POINTS] = {"
    ]

    # Add SoC points (assuming they're the same for all temperature curves)
    soc_points = soc_curves[first_temp]['curve'][0]
    # Split array into chunks of 8 elements for better readability
    for i in range(0, len(soc_points), 8):
        chunk = soc_points[i:i+8]
        header.append("    " + ", ".join([f"{soc:.6f}f" for soc in chunk]) +
                     ("," if i+8 < len(soc_points) else ""))
    header.append("};")
    header.append("")

    # Add V_oc arrays for each temperature
    header.append("// Open circuit voltage arrays for each temperature")
    header.append("static const float BATTERY_VOC_ARRAYS[NUM_TEMPERATURE_POINTS][NUM_SOC_POINTS] = {")

    for temp_idx, temp in enumerate(temp_points):
        header.append(f"    // Temperature: {temp:.2f}°C")
        header.append("    {")
        voc_points = soc_curves[temp]['curve'][1]
        # Split array into chunks of 8 elements for better readability
        for i in range(0, len(voc_points), 8):
            chunk = voc_points[i:i+8]
            header.append("        " + ", ".join([f"{voc:.6f}f" for voc in chunk]) +
                         ("," if i+8 < len(voc_points) else ""))
        header.append("    }" + ("," if temp_idx < len(temp_points)-1 else ""))
    header.append("};")
    header.append("")

    # Add function declarations
    header.extend([
        "// Function declarations",
        "/**",
        " * Get internal resistance at specified temperature",
        " * @param temperature Battery temperature in Celsius",
        " * @return Interpolated internal resistance (ohms)",
        " */",
        "float battery_get_internal_resistance(float temperature);",
        "",
        "/**",
        " * Get battery capacity at specified temperature",
        " * @param temperature Battery temperature in Celsius",
        " * @return Interpolated capacity (mAh)",
        " */",
        "float battery_get_capacity(float temperature);",
        "",
        "/**",
        " * Get open circuit voltage for given SoC and temperature",
        " * @param soc State of Charge (0.0 to 1.0)",
        " * @param temperature Battery temperature in Celsius",
        " * @return Open circuit voltage (V)",
        " */",
        "float battery_get_voc(float soc, float temperature);",
        "",
        "/**",
        " * Get State of Charge for given open circuit voltage and temperature",
        " * @param voc Open circuit voltage (V)",
        " * @param temperature Battery temperature in Celsius",
        " * @return State of Charge (0.0 to 1.0)",
        " */",
        "float battery_get_soc(float voc, float temperature);",
        "",
        "/**",
        " * Calculate OCV-SOC curve slope at given point (for Jacobian in EKF)",
        " * @param soc State of Charge (0.0 to 1.0)",
        " * @param temperature Battery temperature in Celsius",
        " * @return Slope of OCV vs SOC curve (V per SOC unit)",
        " */",
        "float battery_calculate_ocv_slope(float soc, float temperature);",
        "",
        "#endif // BATTERY_LOOKUP_TABLES_H"
    ])

    # Write the header to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(header))

    print(f"Battery lookup table header generated: {output_path}")


def generate_battery_lookup_implementation(soc_curves, output_path="battery_lookup_tables.c"):
    """Generate C implementation file for battery lookup functions"""
    # Get sorted temperature points
    temp_points = sorted(list(soc_curves.keys()))

    # Determine voltage direction
    voc_array = soc_curves[temp_points[0]]['curve'][1]
    voltage_decreases = voc_array[0] > voc_array[-1]

    # Start building the implementation file
    impl = [
        "/**",
        " * Battery SoC Lookup Tables Implementation",
        " * Auto-generated from Python battery characterization script",
        " * Battery type: LiFePO4",
        " */",
        "",
        "#include \"battery_lookup_tables.h\"",
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
        "float battery_get_internal_resistance(float temperature) {",
        "    // Handle out-of-bounds temperatures",
        "    if (temperature <= BATTERY_TEMP_POINTS[0]) {",
        "        return BATTERY_R_INT[0];",
        "    }",
        "    if (temperature >= BATTERY_TEMP_POINTS[NUM_TEMPERATURE_POINTS - 1]) {",
        "        return BATTERY_R_INT[NUM_TEMPERATURE_POINTS - 1];",
        "    }",
        "",
        "    // Find temperature bracket",
        "    for (int i = 0; i < NUM_TEMPERATURE_POINTS - 1; i++) {",
        "        if (temperature < BATTERY_TEMP_POINTS[i + 1]) {",
        "            return linear_interpolate(temperature,",
        "                                      BATTERY_TEMP_POINTS[i], BATTERY_R_INT[i],",
        "                                      BATTERY_TEMP_POINTS[i + 1], BATTERY_R_INT[i + 1]);",
        "        }",
        "    }",
        "",
        "    // Should never reach here",
        "    return BATTERY_R_INT[0];",
        "}",
        "",
        "float battery_get_capacity(float temperature) {",
        "    // Handle out-of-bounds temperatures",
        "    if (temperature <= BATTERY_TEMP_POINTS[0]) {",
        "        return BATTERY_CAPACITY[0];",
        "    }",
        "    if (temperature >= BATTERY_TEMP_POINTS[NUM_TEMPERATURE_POINTS - 1]) {",
        "        return BATTERY_CAPACITY[NUM_TEMPERATURE_POINTS - 1];",
        "    }",
        "",
        "    // Find temperature bracket",
        "    for (int i = 0; i < NUM_TEMPERATURE_POINTS - 1; i++) {",
        "        if (temperature < BATTERY_TEMP_POINTS[i + 1]) {",
        "            return linear_interpolate(temperature,",
        "                                      BATTERY_TEMP_POINTS[i], BATTERY_CAPACITY[i],",
        "                                      BATTERY_TEMP_POINTS[i + 1], BATTERY_CAPACITY[i + 1]);",
        "        }",
        "    }",
        "",
        "    // Should never reach here",
        "    return BATTERY_CAPACITY[0];",
        "}",
        "",
        "// Helper function to get VOC at a specific temperature",
        "static float get_voc_at_temp(float soc, int temp_idx) {",
        "    // Clip SoC to valid range",
        "    if (soc <= BATTERY_SOC_POINTS[0]) {",
        "        return BATTERY_VOC_ARRAYS[temp_idx][0];",
        "    }",
        "    if (soc >= BATTERY_SOC_POINTS[NUM_SOC_POINTS - 1]) {",
        "        return BATTERY_VOC_ARRAYS[temp_idx][NUM_SOC_POINTS - 1];",
        "    }",
        "",
        "    // Find SoC bracket",
        "    for (int i = 0; i < NUM_SOC_POINTS - 1; i++) {",
        "        if (soc < BATTERY_SOC_POINTS[i + 1]) {",
        "            return linear_interpolate(soc,",
        "                                     BATTERY_SOC_POINTS[i], BATTERY_VOC_ARRAYS[temp_idx][i],",
        "                                     BATTERY_SOC_POINTS[i + 1], BATTERY_VOC_ARRAYS[temp_idx][i + 1]);",
        "        }",
        "    }",
        "",
        "    // Should never reach here",
        "    return BATTERY_VOC_ARRAYS[temp_idx][0];",
        "}",
        "",
        "float battery_get_voc(float soc, float temperature) {",
        "    // Handle out-of-bounds temperatures",
        "    if (temperature <= BATTERY_TEMP_POINTS[0]) {",
        "        return get_voc_at_temp(soc, 0);",
        "    }",
        "    if (temperature >= BATTERY_TEMP_POINTS[NUM_TEMPERATURE_POINTS - 1]) {",
        "        return get_voc_at_temp(soc, NUM_TEMPERATURE_POINTS - 1);",
        "    }",
        "",
        "    // Find temperature bracket",
        "    for (int i = 0; i < NUM_TEMPERATURE_POINTS - 1; i++) {",
        "        if (temperature < BATTERY_TEMP_POINTS[i + 1]) {",
        "            float voc_low = get_voc_at_temp(soc, i);",
        "            float voc_high = get_voc_at_temp(soc, i + 1);",
        "",
        "            return linear_interpolate(temperature,",
        "                                     BATTERY_TEMP_POINTS[i], voc_low,",
        "                                     BATTERY_TEMP_POINTS[i + 1], voc_high);",
        "        }",
        "    }",
        "",
        "    // Should never reach here",
        "    return get_voc_at_temp(soc, 0);",
        "}",
        "",
        "// Helper function to get SoC at a specific temperature",
        "static float get_soc_at_temp(float voc, int temp_idx) {",
        "    // Determine min/max voltage based on battery chemistry direction",
        "    const float min_voc = VOC_DECREASES_WITH_SOC ? ",
        "                          BATTERY_VOC_ARRAYS[temp_idx][NUM_SOC_POINTS - 1] : BATTERY_VOC_ARRAYS[temp_idx][0];",
        "    const float max_voc = VOC_DECREASES_WITH_SOC ? ",
        "                          BATTERY_VOC_ARRAYS[temp_idx][0] : BATTERY_VOC_ARRAYS[temp_idx][NUM_SOC_POINTS - 1];",
        "",
        "    // Handle out-of-bounds voltages",
        "    if (voc <= min_voc) {",
        "        return VOC_DECREASES_WITH_SOC ? BATTERY_SOC_POINTS[NUM_SOC_POINTS - 1] : BATTERY_SOC_POINTS[0];",
        "    }",
        "    if (voc >= max_voc) {",
        "        return VOC_DECREASES_WITH_SOC ? BATTERY_SOC_POINTS[0] : BATTERY_SOC_POINTS[NUM_SOC_POINTS - 1];",
        "    }",
        "",
        "    // Find VOC bracket",
        "    if (VOC_DECREASES_WITH_SOC) {",
        "        // VOC decreases as SOC increases (typical for LiFePO4)",
        "        for (int i = 0; i < NUM_SOC_POINTS - 1; i++) {",
        "            if (voc >= BATTERY_VOC_ARRAYS[temp_idx][i + 1]) {",
        "                return linear_interpolate(voc,",
        "                                         BATTERY_VOC_ARRAYS[temp_idx][i], BATTERY_SOC_POINTS[i],",
        "                                         BATTERY_VOC_ARRAYS[temp_idx][i + 1], BATTERY_SOC_POINTS[i + 1]);",
        "            }",
        "        }",
        "    } else {",
        "        // VOC increases as SOC increases",
        "        for (int i = 0; i < NUM_SOC_POINTS - 1; i++) {",
        "            if (voc <= BATTERY_VOC_ARRAYS[temp_idx][i + 1]) {",
        "                return linear_interpolate(voc,",
        "                                         BATTERY_VOC_ARRAYS[temp_idx][i], BATTERY_SOC_POINTS[i],",
        "                                         BATTERY_VOC_ARRAYS[temp_idx][i + 1], BATTERY_SOC_POINTS[i + 1]);",
        "            }",
        "        }",
        "    }",
        "",
        "    // Should never reach here",
        "    return BATTERY_SOC_POINTS[0];",
        "}",
        "",
        "float battery_get_soc(float voc, float temperature) {",
        "    // Handle out-of-bounds temperatures",
        "    if (temperature <= BATTERY_TEMP_POINTS[0]) {",
        "        return get_soc_at_temp(voc, 0);",
        "    }",
        "    if (temperature >= BATTERY_TEMP_POINTS[NUM_TEMPERATURE_POINTS - 1]) {",
        "        return get_soc_at_temp(voc, NUM_TEMPERATURE_POINTS - 1);",
        "    }",
        "",
        "    // Find temperature bracket",
        "    for (int i = 0; i < NUM_TEMPERATURE_POINTS - 1; i++) {",
        "        if (temperature < BATTERY_TEMP_POINTS[i + 1]) {",
        "            float soc_low = get_soc_at_temp(voc, i);",
        "            float soc_high = get_soc_at_temp(voc, i + 1);",
        "",
        "            return linear_interpolate(temperature,",
        "                                     BATTERY_TEMP_POINTS[i], soc_low,",
        "                                     BATTERY_TEMP_POINTS[i + 1], soc_high);",
        "        }",
        "    }",
        "",
        "    // Should never reach here",
        "    return get_soc_at_temp(voc, 0);",
        "}",
        "",
        "float battery_calculate_ocv_slope(float soc, float temperature) {",
        "    // Use numerical differentiation to calculate slope",
        "    const float delta = 0.01f;  // Small delta for differentiation",
        "",
        "    // Make sure we don't go out of bounds",
        "    float soc_plus = (soc + delta > 1.0f) ? 1.0f : (soc + delta);",
        "    float soc_minus = (soc - delta < 0.0f) ? 0.0f : (soc - delta);",
        "",
        "    // Calculate OCVs at nearby points",
        "    float voc_plus = battery_get_voc(soc_plus, temperature);",
        "    float voc_minus = battery_get_voc(soc_minus, temperature);",
        "",
        "    // Calculate slope via central difference",
        "    float slope = (voc_plus - voc_minus) / (soc_plus - soc_minus);",
        "    ",
        "    return slope;",
        "}"
    ]

    # Write the implementation to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(impl))

    print(f"Battery lookup table implementation generated: {output_path}")


def export_c_battery_lookup(soc_curves, output_dir="."):
    """Export battery lookup tables to C files"""
    # Create directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)

    generate_battery_lookup_header(soc_curves, f"{output_dir}/battery_lookup_tables.h")
    generate_battery_lookup_implementation(soc_curves, f"{output_dir}/battery_lookup_tables.c")
    print(f"Battery lookup tables exported to C files in {output_dir}")