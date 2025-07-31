
import matplotlib.pyplot as plt
import numpy as np
from .battery_model import BatteryModel

class DummyEstimator():

    def __init__(self,
                 battery_model : BatteryModel):

        # Batery model consis of characterized battery data
        self.bm = battery_model
        self.reset()

    def reset(self):

        # Reset default state
        self.x = 0 # SoC
        self.x_latched = self.x


    def initial_guess(self, voltage_V, current_mA, temp_deg):
        """
        Use the very first measurement to initialize the state of charge
        just by interpolation on the SoC curve.
        """

        discharge_mode = True
        if(current_mA < 0):
            discharge_mode = False

        ocv = self.bm._meas_to_ocv(voltage_V, current_mA, temp_deg)
        self.x = self.bm._interpolate_soc_at_temp(ocv, temp_deg, discharge_mode)
        self.x_latched = self.x

        return

    def update(self, dt, voltage_V, current_mA, temp_deg):

        discharge_mode = True
        if(current_mA < 0):
            discharge_mode = False

        ocv = self.bm._meas_to_ocv(voltage_V, current_mA, temp_deg)
        self.x = self.bm._interpolate_soc_at_temp(ocv, temp_deg, discharge_mode)
        self.x_latched = self.x

        return self.x_latched, None
