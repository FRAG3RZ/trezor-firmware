
import matplotlib.pyplot as plt
import numpy as np


class fuel_gauge():

    model_types = ["ekf", "ekf_adaptive", "ukf", "direct"]

    def __init__(self, soc_curve,  model_type=None, R=None, Q=None, R_agressive=None, Q_agressive=None, P_init=None, n=None, alpha=None, beta=None, kappa=None):

        if model_type is not None:
            if model_type not in self.model_types:
                raise ValueError(f"model_type should be one of the following: {self.model_types}")
            self.model_type = model_type

        self.soc_curve = soc_curve # curve should be a list of tuples (SoC, V_oc) in ascending order
        self.temp_keys_list = sorted(list(soc_curve.keys()))

        if model_type == "direct":
            # No parameters needed
            pass

        if model_type == "ekf" or model_type == "ekf_adaptive":
            if R is None or Q is None or P_init is None:
                raise ValueError("Some of the EKF parameters are missing")

            self.R_default = R
            self.Q_default = Q
            self.R_agressive_default = R_agressive
            self.Q_agressive_default = Q_agressive
            self.P_init_default = P_init

        if model_type == "ukf":
            if R is None or Q is None or n is None or alpha is None or beta is None or kappa is None or P_init is None:
                raise ValueError("Some of the UKF parameters are missing")

            self.R_default = R
            self.Q_default = Q
            self.n_default = n              # State dimension
            self.alpha_default = alpha      # Controls spread of sigma points (small value for highly nonlinear systems)
            self.beta_default = beta        # Optimal for Gaussian distributions
            self.kappa_default = kappa      # Secondary scaling parameter
            self.P_init_default = P_init

        self.reset()


    def reset(self):

        if self.model_type == "direct":
            pass

        if self.model_type == "ekf" or self.model_type == "ekf_adaptive":
            self.R = self.R_default
            self.Q = self.Q_default
            self.P_init = self.P_init_default
            self.P = self.P_init_default

            self.v_meas_history = []
            self.i_meas_history = []
            self.filter_window = 5  # Number of samples to average

        if self.model_type == "ukf":
            self.R = self.R_default
            self.Q = self.Q_default
            self.n = self.n_default
            self.alpha = self.alpha_default
            self.beta = self.beta_default
            self.kappa = self.kappa_default
            self.lambda_ukf = self.alpha**2 * (self.n + self.kappa) - self.n
            self.P = self.P_init_default

            # Calculate weights
            self.weights_mean = np.zeros(2*self.n + 1)
            self.weights_cov = np.zeros(2*self.n + 1)

            # Weight for mean and covariance of central point
            self.weights_mean[0] = self.lambda_ukf / (self.n + self.lambda_ukf)
            self.weights_cov[0] = self.weights_mean[0] + (1 - self.alpha**2 + self.beta)

            # Weights for remaining sigma points
            for i in range(1, 2*self.n + 1):
                self.weights_mean[i] = 1 / (2 * (self.n + self.lambda_ukf))
                self.weights_cov[i] = self.weights_mean[i]

            self.v_meas_history = []
            self.i_meas_history = []
            self.filter_window = 5  # Number of samples to average

        # Reset default state
        self.x = 0 # SoC
        self.x_latched = self.x

    def _filter_measurement(self, new_value, history):
        """Simple moving average filter for measurements"""
        history.append(new_value)
        if len(history) > self.filter_window:
            history.pop(0)

        return sum(history) / len(history)

    def run_simulation(self, waveform, sp=0, override_init_soc=None, init_filter=False):

        SoC = np.zeros((len(waveform.time)))
        covariance = np.zeros((len(waveform.time)))
        time = np.zeros((len(waveform.time)))
        termination_voc = np.zeros((len(waveform.time)))
        voc = np.zeros((len(waveform.time)))

        self.reset()

        if init_filter:
            init_filter_len = 15
        else:
            init_filter_len = 1

        self.initial_guess(waveform.vbat[sp:(sp+init_filter_len)], waveform.ibat[sp:(sp+init_filter_len)], waveform.ntc_temp[sp:(sp+init_filter_len)], override_init_soc)
        sim_start = sp + init_filter_len

        termination_voc[0] = waveform.vbat[0]

        sim_end = 0

        for i in range(sp, len(waveform.time)):

            if i == 0:
                continue

            time[i] = waveform.time[i]

            if self.model_type == "ukf":
                SoC[i], _ = self.update_ukf(waveform.time[i]-waveform.time[i-1], waveform.vbat[i], waveform.ibat[i], waveform.ntc_temp[i])
            elif self.model_type == "ekf":
                SoC[i], _ = self.update_ekf(waveform.time[i]-waveform.time[i-1], waveform.vbat[i], waveform.ibat[i], waveform.ntc_temp[i])
            elif self.model_type == "direct":
                SoC[i] = self.simple_guess(waveform.vbat[i], waveform.ibat[i], waveform.ntc_temp[i])
            elif self.model_type == "ekf_adaptive":
                SoC[i], _ = self.update_ekf_adaptive(waveform.time[i]-waveform.time[i-1], waveform.vbat[i], waveform.ibat[i], waveform.ntc_temp[i])
            sim_end = i

        return time, SoC, sim_start, sim_end

    def _adapt_filter_params(self, I_meas, T_meas):
        """Adapt Kalman filter parameters based on operating conditions"""
        # Adjust process noise based on current magnitude (higher current = higher uncertainty)
        current_abs = abs(I_meas)
        if current_abs > 100:  # High current
            self.Q = 0.0003  # Higher process noise during high currents
        else:
            self.Q = 0.00005  # Lower process noise during low currents/rest

        # Adjust measurement noise based on temperature (extreme temps = higher uncertainty)
        if T_meas < 15 or T_meas > 30:
            self.R = 5000  # Higher measurement noise at extreme temperatures
        else:
            self.R = 100  # Lower measurement noise at moderate temperatures

    def simple_guess(self, V_meas, I_meas, T_meas):

        V_oc = self._meas_to_ocv(V_meas, I_meas, T_meas)

        soc_int = self._interpolate_soc_at_temp(V_oc, T_meas)

        return soc_int

    def initial_guess(self, V_meas, I_meas, T_meas, override_init_soc=None):
        """simple
        Use the very first measurement to initialize the state of charge
        just by interpolation on the SoC curve.
        """

        # V_oc = 0
        # T_oc = 0
        # for i in range(0, len(V_meas)):
        #     V_oc += self._meas_to_ocv(V_meas[i], I_meas[i], T_meas[i])
        #     T_oc += T_meas[i]
        # V_oc = V_oc / len(V_meas)
        # T_oc = T_oc / len(T_meas)

        V_oc = self._meas_to_ocv(V_meas[0], I_meas[0], T_meas[0])
        T_oc = T_meas[0]

        self.x = self._interpolate_soc_at_temp(V_oc, T_oc)
        self.x_latched = self.x

        if override_init_soc is not None:
            self.x = override_init_soc
            self.x_latched = self.x

        if(self.model_type == "ukf" or self.model_type == "ekf"):
            self.P = self.P_init_default

        return

    def update(self, dt,  V_meas, I_meas, T_meas):
        """
        Use the measurement of battery terminal voltage, current and temperature
        to estimate the state of charge
        """

        V_meas = self._filter_measurement(V_meas, self.v_meas_history)
        I_meas = self._filter_measurement(I_meas, self.i_meas_history)

        #self._adapt_filter_params(I_meas, T_meas)

        # dt got to be in seconds
        x_k1_k = self.x - (I_meas/(3600*self._total_capacity(T_meas)))*(dt/1000)

        P_k1_k = self.P + self.Q
        K_k1_k = P_k1_k / (P_k1_k + self.R) # Here the jacobian is being neglected
        x_k1_k1 = x_k1_k + K_k1_k*(V_meas - (self._interpolate_voc_at_temp(x_k1_k, T_meas) - (I_meas/1000)*self._rint(T_meas)))
        P_k1_k1 = (1 - K_k1_k)*P_k1_k

        self.x = x_k1_k1
        self.P = P_k1_k1

        return self.x, self.P


    def update_ekf_adaptive(self, dt, V_meas, I_meas, T_meas):

        V_meas = self._filter_measurement(V_meas, self.v_meas_history)
        I_meas = self._filter_measurement(I_meas, self.i_meas_history)

        if(T_meas < 10):
            self.R = 10 #self._linear_interpolation(10, 1, 10, -20, T_meas)
            self.Q = 0.01 #self._linear_interpolation(self.Q_default, self.Q_agressive_default, 10, -20, T_meas)
        else:

            # Pick fuel gauge agressivity
            if self.x_latched < 0.2:
                self.R = self.R_agressive_default
                self.Q = self.Q_agressive_default
            else:
                self.R = self.R_default
                self.Q = self.Q_default

        #self._adapt_filter_params(I_meas, T_meas)
        # Convert dt to seconds
        dt_sec = dt / 1000.0

        # State prediction (coulomb counting)
        x_k1_k = self.x - (I_meas/(3600*self._total_capacity(T_meas)))*dt_sec

        # Calculate Jacobian of measurement function h(x) with respect to x
        # For the battery model: h(x) = OCV(x) - R*I
        # So h'(x) = dOCV/dx
        # h_jacobian = self._calculate_ocv_slope(x_k1_k, T_meas)
        h_jacobian = 1

        # Error covariance prediction
        P_k1_k = self.P + self.Q

        # Calculate innovation covariance
        S = h_jacobian * P_k1_k * h_jacobian + self.R

        # Calculate Kalman gain
        K_k1_k = P_k1_k * h_jacobian / S

        # Calculate predicted terminal voltage
        v_pred = self._interpolate_voc_at_temp(x_k1_k, T_meas) - (I_meas/1000)*self._rint(T_meas)

        # State update
        x_k1_k1 = x_k1_k + K_k1_k * (V_meas - v_pred)

        # Error covariance update
        P_k1_k1 = (1 - K_k1_k * h_jacobian) * P_k1_k

        # Enforce SoC boundaries
        self.x = max(0.0, min(1.0, x_k1_k1))
        self.P = P_k1_k1

        # Based on the current directon decide what to latch
        if(I_meas > 0):
            # Discharging, Soc should move only in negative direction
            if(self.x < self.x_latched):
                self.x_latched = self.x
        else:
            if(self.x > self.x_latched):
                self.x_latched = self.x

        return self.x_latched, self.P

    def update_ekf(self, dt, V_meas, I_meas, T_meas):

        V_meas = self._filter_measurement(V_meas, self.v_meas_history)
        I_meas = self._filter_measurement(I_meas, self.i_meas_history)

        #self._adapt_filter_params(I_meas, T_meas)
        # Convert dt to seconds
        dt_sec = dt / 1000.0

        # State prediction (coulomb counting)
        x_k1_k = self.x - (I_meas/(3600*self._total_capacity(T_meas)))*dt_sec

        # Calculate Jacobian of measurement function h(x) with respect to x
        # For the battery model: h(x) = OCV(x) - R*I
        # So h'(x) = dOCV/dx
        # h_jacobian = self._calculate_ocv_slope(x_k1_k, T_meas)
        h_jacobian = 1

        # Error covariance prediction
        P_k1_k = self.P + self.Q

        # Calculate innovation covariance
        S = h_jacobian * P_k1_k * h_jacobian + self.R

        # Calculate Kalman gain
        K_k1_k = P_k1_k * h_jacobian / S

        # Calculate predicted terminal voltage
        v_pred = self._interpolate_voc_at_temp(x_k1_k, T_meas) - (I_meas/1000)*self._rint(T_meas)

        # State update
        x_k1_k1 = x_k1_k + K_k1_k * (V_meas - v_pred)

        # Error covariance update
        P_k1_k1 = (1 - K_k1_k * h_jacobian) * P_k1_k

        # Enforce SoC boundaries
        self.x = max(0.0, min(1.0, x_k1_k1))
        self.P = P_k1_k1



        return self.x, self.P

    def _calculate_ocv_slope(self, soc, temp):
        """Calculate the slope (derivative) of the OCV-SOC curve at a given SoC and temperature"""

        delta = 0.01  # Small delta for numerical differentiation

        # Make sure we don't go out of bounds
        soc_plus = min(1.0, soc + delta)
        soc_minus = max(0.0, soc - delta)

        # Calculate OCVs at nearby points
        ocv_plus = self._interpolate_voc_at_temp(soc_plus, temp)
        ocv_minus = self._interpolate_voc_at_temp(soc_minus, temp)

        # Calculate slope via central difference
        slope = (ocv_plus - ocv_minus) / (soc_plus - soc_minus)
        return slope

    def get_SoC(self):
        return self.x

    def _meas_to_ocv(self, V_meas, I_meas, T_meas):
        rint = self._rint(T_meas)
        Voc = V_meas + ((I_meas/1000) * rint)
        return Voc

    def _rint(self, temp):

        temp = max(min(temp, self.temp_keys_list[-1]), self.temp_keys_list[0])

        # Search closest
        dist = 1000
        closest_temp = 0
        for i, list_temp in enumerate(self.temp_keys_list):
            if(abs(temp - list_temp) < dist):
                dist = abs(temp - list_temp)
                closest_temp = list_temp

        for i, curve_temp in enumerate(self.temp_keys_list):

            if(curve_temp >= temp):
                # Linear interpolation
                t2 = curve_temp
                t1 = self.temp_keys_list[i-1]
                R2 = self.soc_curve[t2]['r_int']
                R1 = self.soc_curve[t1]['r_int']

                R_int_interpolated = self._linear_interpolation(R1, R2, t1, t2, temp)

                return R_int_interpolated

    def _total_capacity(self, temp):

        temp = max(min(temp, self.temp_keys_list[-1]), self.temp_keys_list[0])
        for i, curve_temp in enumerate(self.temp_keys_list):

            if(curve_temp >= temp):
                # Linear interpolation
                t2 = curve_temp
                t1 = self.temp_keys_list[i-1]
                AH2 = self.soc_curve[t2]['total_capacity']
                AH1 = self.soc_curve[t1]['total_capacity']
                return self._linear_interpolation(AH1, AH2, t1, t2, temp)

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

    def _linear_interpolation_Voc(self, voc1, voc2, soc1, soc2, x):
        a = (x2[1] - x1[1]) / (x2[0] - x1[0])
        b = x1[1] - a*x1[0]
        return a*x + b

    def _linear_interpolation_SoC(self, x1,x2, x):
        a = (x2[0] - x1[0]) / (x2[1] - x1[1])
        b = x1[0] - a*x1[1]
        return a*x + b

    def _interpolate_voc_at_temp(self, soc, temp):

        temp = max(min(temp, self.temp_keys_list[-1]), self.temp_keys_list[0])

        for i, curve_temp in enumerate(self.temp_keys_list):

            if(curve_temp >= temp):
                # Linear interpolation
                t2 = curve_temp
                t1 = self.temp_keys_list[i-1]
                voc2 = self._interpolate_voc(self.soc_curve[t2]['curve'], soc)
                voc1 = self._interpolate_voc(self.soc_curve[t1]['curve'], soc)
                return self._linear_interpolation(voc1, voc2, t1, t2, temp)

        pass

    def _interpolate_soc_at_temp(self, voc, temp):

        temp = max(min(temp, self.temp_keys_list[-1]), self.temp_keys_list[0])

        for i, curve_temp in enumerate(self.temp_keys_list):

            if(curve_temp >= temp):

                # Linear interpolation
                t2 = curve_temp
                t1 = self.temp_keys_list[i-1]

                soc2 = self._interpolate_soc(self.soc_curve[t2]['curve'], voc)
                soc1 = self._interpolate_soc(self.soc_curve[t1]['curve'], voc)

                soc_inter = self._linear_interpolation(soc1, soc2, t1, t2, temp)

                return soc_inter

        pass

    def _interpolate_voc(self,soc_curve, soc):

        soc = max(min(soc,soc_curve[0][-1]),soc_curve[0][0])

        for i in range(0, len(soc_curve[0])):

            if(i == 0):
                continue

            if(soc > soc_curve[0][i-1] and soc <= soc_curve[0][i]):
                soc1 = soc_curve[0][i-1]
                soc2 = soc_curve[0][i]
                voc1 = soc_curve[1][i-1]
                voc2 = soc_curve[1][i],
                return self._linear_interpolation(voc1, voc2, soc1, soc2, soc)

        return soc_curve[1][0] if soc <= soc_curve[0][0] else soc_curve[1][-1]

    def _interpolate_soc(self, soc_curve, voc):

        voc = max(min(voc, soc_curve[1][-1]), soc_curve[1][0])

        for i in range(0, len(soc_curve[1])):

            if(i == 0):
                continue

            if(voc > soc_curve[1][i-1] and voc <= soc_curve[1][i]):
                soc1 = soc_curve[0][i-1]
                soc2 = soc_curve[0][i]
                voc1 = soc_curve[1][i-1]
                voc2 = soc_curve[1][i]
                return self._linear_interpolation(soc1, soc2, voc1, voc2, voc)

        return soc_curve[0][0] if voc <= soc_curve[1][0] else soc_curve[0][-1]

    def update_ukf(self, dt, V_meas, I_meas, T_meas):
        """UKF update for battery SoC estimation"""
        # Filter measurements
        #V_meas = self._filter_measurement(V_meas, self.v_meas_history)
        #I_meas = self._filter_measurement(I_meas, self.i_meas_history)

        # Adapt parameters based on operating conditions
        #self._adapt_filter_params(I_meas, T_meas)

        # Convert time to seconds
        dt_sec = dt / 1000.0

        # 1. Generate sigma points around current state estimate
        sigma_points = self._generate_sigma_points()

        # 2. Time update (prediction) - propagate sigma points through process model
        sigma_points_pred = []

        for sp in sigma_points:
            # Apply battery model (coulomb counting) to each sigma point
            sp_new = sp - (I_meas/(3600*self._total_capacity(T_meas)))*dt_sec
            sigma_points_pred.append(sp_new)

        # 3. Calculate predicted state and covariance
        x_pred = 0
        for i in range(len(sigma_points_pred)):
            x_pred += self.weights_mean[i] * sigma_points_pred[i]

        P_pred = self.Q  # Start with process noise
        for i in range(len(sigma_points_pred)):
            P_pred += self.weights_cov[i] * ((sigma_points_pred[i] - x_pred)**2)

        # 4. Measurement update - transform sigma points through measurement model
        y_pred = []
        for sp in sigma_points_pred:
            # Predicted terminal voltage for each sigma point
            v_terminal = self._interpolate_voc_at_temp(sp, T_meas) - (I_meas/1000)*self._rint(T_meas)
            y_pred.append(v_terminal)

        # 5. Calculate predicted measurement mean
        y_mean = 0
        for i in range(len(y_pred)):
            y_mean += self.weights_mean[i] * y_pred[i]

        # 6. Calculate innovation covariance and cross-correlation
        Pyy = self.R  # Start with measurement noise
        Pxy = 0

        for i in range(len(sigma_points_pred)):
            Pyy += self.weights_cov[i] * ((y_pred[i] - y_mean)**2)
            Pxy += self.weights_cov[i] * (sigma_points_pred[i] - x_pred) * (y_pred[i] - y_mean)

        # 7. Calculate Kalman gain
        K = Pxy / Pyy

        # 8. Update state and covariance
        x_new = x_pred + K * (V_meas - y_mean)

        P_new = P_pred - K**2 * Pyy

        # 9. Enforce state constraints
        self.x = max(0.0, min(1.0, x_new))
        self.P = P_new

        return self.x, self.P

    def _generate_sigma_points(self):
        """Generate sigma points around current state estimate"""
        sigma_points = [self.x]

        # Calculate square root of covariance matrix (scalar in this case)
        sqrt_P = np.sqrt(self.P * (self.n + self.lambda_ukf))

        # Generate positive and negative sigma points
        sigma_points.append(self.x + sqrt_P)
        sigma_points.append(self.x - sqrt_P)

        return sigma_points








