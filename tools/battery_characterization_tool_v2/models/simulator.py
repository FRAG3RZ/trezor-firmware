import numpy as np
from utils.error_functions import compute_soc_error, compute_soc_end_error, compute_linear_deviation_error
class SimulationResult:
    def __init__(self, time, soc, covariance, start_idx, end_idx, model_name=None, error=None, error_soc=None, deviation_error=None):
        self.time: np.ndarray = time
        self.soc: np.ndarray = soc
        self.covariance: np.ndarray = covariance
        self.start_idx: int = start_idx
        self.end_idx: int = end_idx
        self.model_name: str = model_name
        self.error: float | None = error       # Total SoC error
        self.error_soc: float | None = error_soc  # End-point SoC error
        self.deviation_error: float | None = deviation_error  # Deviation error from the reference model


def mean_filter(data):
    return sum(data) / len(data)


def run_battery_simulation(
    waveform,
    soc_estim_model,
    sim_start_idx=0,
    initial_soc=None,
    init_filter_length=10,
    battery_model=None,
    sim_name = None
):
    soc = np.zeros((len(waveform.time)))
    covariance = np.zeros((len(waveform.time)))

    if sim_start_idx >= len(waveform.time):
        raise ValueError(
            "simulation start index is greater than the length of the waveform"
        )

    # Reset Estimator to default
    soc_estim_model.reset()

    if initial_soc is not None:
        soc_estim_model.set_soc(initial_soc)
    else:
        vbat_init = mean_filter(
            waveform.vbat[sim_start_idx : sim_start_idx + init_filter_length]
        )
        ibat_init = mean_filter(
            waveform.ibat[sim_start_idx : sim_start_idx + init_filter_length]
        )
        ntc_temp_init = mean_filter(
            waveform.ntc_temp[sim_start_idx : sim_start_idx + init_filter_length]
        )

        soc_estim_model.initial_guess(vbat_init, ibat_init, ntc_temp_init)

    sim_end_idx = 0
    for i in range(sim_start_idx + 1 + init_filter_length, len(waveform.time)):
        soc[i], covariance[i] = soc_estim_model.update(
            waveform.time[i] - waveform.time[i - 1],
            waveform.vbat[i],
            waveform.ibat[i],
            waveform.ntc_temp[i],
        )
        sim_end_idx = i

    # ==== Compute errors if battery_model provided ====
    soc_err = None
    end_soc_err = None
    deviation_error = None
    if battery_model is not None:
        soc_err = compute_soc_error(
            waveform,
            soc,
            sim_start_idx + 1 + init_filter_length,
            sim_end_idx,
            battery_model,
            sim_name=sim_name,
        ) or None
        end_soc_err = compute_soc_end_error(
            waveform,
            soc,
            sim_name=sim_name,
        ) or None
        deviation_error = compute_linear_deviation_error(
            waveform,
            soc,
            sim_start_idx + 1 + init_filter_length,
            sim_end_idx,
            sim_name=sim_name
        ) or None

    # ====================================================================

    sim_result = SimulationResult(
        waveform.time, soc, covariance, sim_start_idx, sim_end_idx, soc_estim_model.name
    )

    #Error appending
    sim_result.error = soc_err
    sim_result.error_soc = end_soc_err
    sim_result.deviation_error = deviation_error

    return sim_result


