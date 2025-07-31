
import numpy as np
import hashlib
import json
import os


class BatteryModel():
   
    def __init__(self, battery_model_data, file_name_hash, override_hash=None):
        self.battery_model_data = battery_model_data
        self.temp_keys_list = sorted([float(t) for t in self.battery_model_data['ocv_curves'].keys()])
        self.soc_breakpoint_1 = 0.25
        self.soc_breakpoint_2 = 0.8

        if override_hash is not None:
            self.model_hash = override_hash
        else:
            self.model_hash = self._generate_hash(file_name_hash)

        print(f"Battery model + file names hash: {self.model_hash}")


    def _generate_hash(self, file_name_hash):
        print(f"File name hash: {file_name_hash}")  # debug print

        def convert_np(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        def recursive_sort(obj):
            if isinstance(obj, dict):
                return {k: recursive_sort(v) for k, v in sorted(obj.items())}
            elif isinstance(obj, list):
                return sorted(recursive_sort(x) for x in obj)
            else:
                return obj

        def round_floats(obj, precision=8):
            if isinstance(obj, dict):
                return {k: round_floats(v, precision) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [round_floats(x, precision) for x in obj]
            elif isinstance(obj, float):
                return round(obj, precision)
            else:
                return obj

        # Clean data for stable serialization
        clean_data = round_floats(recursive_sort(self.battery_model_data))

        # Use pretty-printing with indentation for consistent hashing
        data_str = json.dumps(clean_data, sort_keys=True, default=convert_np, indent=2)

        print("DEBUG JSON string (full):")
        print(data_str)  # print entire JSON string, pretty-printed

        data_hash = hashlib.sha256(data_str.encode('utf-8')).hexdigest()

        print(f"DEBUG Data hash: {data_hash}")  # debug print

        combined = f"{data_hash}|{file_name_hash}"
        final_hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()
        return final_hash[:8]

    @classmethod
    def from_json(cls, json_dict, model_hash):
        """Used when loading an existing model from disk."""
        return cls(json_dict, file_name_hash=model_hash, override_hash=model_hash)

    
    #========== Public methods for battery model ==========

    def _meas_to_ocv(self, voltage_V, current_mA, temp_deg):
        ocv_V = voltage_V + ((current_mA/1000) * self._rint(temp_deg))
        return ocv_V

    def _rint(self, temp_deg):

        temp_deg = max(min(temp_deg, self.temp_keys_list[-1]),
                       self.temp_keys_list[0])

        [a, b, c, d] = self.battery_model_data['r_int']
        return (a + b*temp_deg) / (c + d*temp_deg)

    def _total_capacity(self, temp_deg, discharging_mode):

        temp_deg = max(min(temp_deg, self.temp_keys_list[-1]),
                       self.temp_keys_list[0])

        ocv_curves = self.battery_model_data['ocv_curves']

        for i, curve_temp in enumerate(self.temp_keys_list):

            if(curve_temp >= temp_deg):
                # Linear interpolation
                t2 = curve_temp
                t1 = self.temp_keys_list[i-1]
                if discharging_mode:
                    AH2 = ocv_curves[str(t2)]['total_capacity_discharge_mean']
                    AH1 = ocv_curves[str(t1)]['total_capacity_discharge_mean']
                else:
                    AH2 = ocv_curves[str(t2)]['total_capacity_charge_mean']
                    AH1 = ocv_curves[str(t1)]['total_capacity_charge_mean']

                return self._linear_interpolation(AH1, AH2, t1, t2, temp_deg)

    def _linear_interpolation(self, y1, y2, x1, x2, x):
        """
        Linear interpolation between two points and given x between them.
        (x1,y1) - First known point on the line
        (x2,y2) - Secodnf known point on the line
        x - Interpolated value, following rule have to apply (x1 < x < x2)
        """
        a = (y2-y1)/(x2-x1)
        b = y2 - a*x2
        return a*x + b

    def _interpolate_ocv_at_temp(self, soc, temp,discharge_mode):

        temp = max(min(temp, self.temp_keys_list[-1]), self.temp_keys_list[0])

        for i, curve_temp in enumerate(self.temp_keys_list):

            if(curve_temp >= temp):
                # Linear interpolation
                t2 = curve_temp
                t1 = self.temp_keys_list[i-1]
                voc2 = self._ocv(self.battery_model_data['ocv_curves'][str(t2)], soc, discharge_mode)
                voc1 = self._ocv(self.battery_model_data['ocv_curves'][str(t1)], soc, discharge_mode)
                return self._linear_interpolation(voc1, voc2, t1, t2, temp)

        pass

    def _interpolate_soc_at_temp(self, ocv, temp, discharge_mode):

        temp = max(min(temp, self.temp_keys_list[-1]), self.temp_keys_list[0])

        ocv_curves = self.battery_model_data['ocv_curves']

        for i, curve_temp in enumerate(self.temp_keys_list):

            if(curve_temp >= temp):

                # Linear interpolation
                t2 = curve_temp
                t1 = self.temp_keys_list[i-1]

                soc2 = self._soc(ocv_curves[str(t2)], ocv, discharge_mode)
                soc1 = self._soc(ocv_curves[str(t1)], ocv, discharge_mode)

                soc_inter = self._linear_interpolation(soc2, soc1, t2, t1, temp)

                return soc_inter

        pass

    def _intrepolate_ocv_slope_at_temp(self, soc, temp, discharge_mode):
        """
        Calculate the slope of the SOC curve at a given SOC and temperature.
        The slope is calculated as the derivative of the SOC function.
        The derivative is piecewise defined, so we need to check which
        segment the SOC falls into and calculate the slope accordingly.
        """

        temp = max(min(temp, self.temp_keys_list[-1]), self.temp_keys_list[0])

        ocv_curves = self.battery_model_data['ocv_curves']

        for i, curve_temp in enumerate(self.temp_keys_list):

            if(curve_temp >= temp):
                # Linear interpolation
                t2 = curve_temp
                t1 = self.temp_keys_list[i-1]

                slope2 = self._ocv_slope(ocv_curves[str(t2)], soc, discharge_mode)
                slope1 = self._ocv_slope(ocv_curves[str(t1)], soc, discharge_mode)

                return self._linear_interpolation(slope2, slope1, t2, t1, temp)

        pass

    def _ocv(self, ocv_curve, soc, discharge_mode):

        soc = max(min(soc, 1), 0)

        if(discharge_mode):
            [m, b, a1, b1, c1, d1, a3, b3, c3, d3] = ocv_curve['ocv_discharge']
        else:
            [m, b, a1, b1, c1, d1, a3, b3, c3, d3] = ocv_curve['ocv_charge']

        if(soc < self.soc_breakpoint_1):
            # First segment (rational)
            return (a1 + b1*soc) / (c1 + d1*soc)
        elif(soc >= self.soc_breakpoint_1 and soc <= self.soc_breakpoint_2):
            # Middle segment (linear)
            return m*soc + b
        elif(soc > self.soc_breakpoint_2):
            # Third segment (rational)
            return (a3 + b3*soc) / (c3 + d3*soc)

        raise ValueError("SOC if out of range")


    def _ocv_slope(self, ocv_curve, soc, discharge_mode):
        """
        Calculate the slope of the OCV curve at a given SOC.
        The slope is calculated as the derivative of the OCV function.
        The derivative is piecewise defined, so we need to check which
        segment the SOC falls into and calculate the slope accordingly.
        """

        if(discharge_mode):
            [m, b, a1, b1, c1, d1, a3, b3, c3, d3] = ocv_curve['ocv_discharge']
        else:
            [m, b, a1, b1, c1, d1, a3, b3, c3, d3] = ocv_curve['ocv_charge']


        if(soc < self.soc_breakpoint_1):
            # First segment (rational)
            return (b1*c1 - a1*d1) / ((c1 + d1*soc)**2)
        elif(soc >= self.soc_breakpoint_1 and soc <= self.soc_breakpoint_2):
            # Middle segment (linear)
            return m
        elif(soc > self.soc_breakpoint_2):
            # Third segment (rational)
            return (b3*c3 - a3*d3) / ((c3 + d3*soc)**2)
        raise ValueError("SOC is out of range")


    def _soc(self, ocv_curve, ocv, discharge_mode):

        ocv_breakpoint_1 = self._ocv(ocv_curve, self.soc_breakpoint_1, discharge_mode)
        ocv_breakpoint_2 = self._ocv(ocv_curve, self.soc_breakpoint_2, discharge_mode)

        if(discharge_mode):
            [m, b, a1, b1, c1, d1, a3, b3, c3, d3] = ocv_curve['ocv_discharge']
        else:
            [m, b, a1, b1, c1, d1, a3, b3, c3, d3] = ocv_curve['ocv_charge']

        if(ocv < ocv_breakpoint_1):
            # First segment (rational)
            return (a1 - c1*ocv)/(d1*ocv - b1)
        elif(ocv >= ocv_breakpoint_1 and ocv <= ocv_breakpoint_2):
            # Middle segment (linear)
            return (ocv - b)/m
        elif(ocv > ocv_breakpoint_2):
            # Third segment (rational)
            return (a3 - c3*ocv)/(d3*ocv - b3)

        raise ValueError("OCV is out of range")




#========FUNCTIONS FOR JSON SERIALIZATION==========

def round_floats(obj, precision=8):
    if isinstance(obj, dict):
        return {k: round_floats(v, precision) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(x, precision) for x in obj]
    elif isinstance(obj, float):
        return round(obj, precision)
    else:
        return obj

def recursive_sort(obj):
    if isinstance(obj, dict):
        return {k: recursive_sort(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        return [recursive_sort(x) for x in obj]  # DO NOT SORT LISTS
    else:
        return obj

def convert_np(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def prepare_for_serialization(data):
    """Sort, round, and convert numpy for consistent hashing/serialization."""
    return recursive_sort(round_floats(data))


def save_battery_model_to_json(battery_model, directory):
    """
    Saves the BatteryModel to a JSON file named <model_hash>.json in the specified directory,
    using fully sorted and rounded data for integrity.
    """
    os.makedirs(directory, exist_ok=True)
    file_name = f"{battery_model.battery_model_data["battery_vendor"]}_{battery_model.model_hash}.json"
    file_path = os.path.join(directory, file_name)

    clean_data = prepare_for_serialization(battery_model.battery_model_data)

    with open(file_path, 'w') as f:
        json.dump(clean_data, f, indent=2, default=convert_np)

    print(f"Battery model saved as: {file_path}")
    return battery_model.model_hash


def load_battery_model_from_hash(battery_manufacturer, model_hash, directory):
    """
    Loads a BatteryModel instance from a JSON file named <model_hash>.json in the specified directory,
    preparing the loaded data identically to preserve hash integrity.
    """
    file_name = f"{model_hash}.json"
    file_path = os.path.join(directory, file_name)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No battery model file found for hash: {model_hash}")

    with open(file_path, 'r') as f:
        battery_model_data = json.load(f)

    # Prepare data to ensure the same hash calculation (if needed)
    battery_model_data = prepare_for_serialization(battery_model_data)

    # Pass the hash string (without '.json') as the second argument
    return BatteryModel.from_json(battery_model_data, model_hash)

