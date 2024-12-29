from dataclasses import dataclass
from pathlib import Path
from typing import Dict

@dataclass(frozen=True)
class WesadDataIngestionConfig:
    data_source_path: Path
    data_target_path: Path
    valid_subjects: list
    required_signals: list
    sampling_rates : Dict[str,Dict[str,int]]
    common_sampling_rate: int
    window_size: int


@dataclass(frozen=True)
class SignalProcessingConfig:
    # Signal filtering
    bvp_filter_low : float
    bvp_filter_high : float
    eda_filter_low : float
    eda_filter_high : float

    quality_threshold:float
    physiological_ranges : Dict[str,Dict[str,float]]

    # Motion artifact
    acc_threshold: float
    motion_window : int

    # window Parameters
    common_sampling_rate: int
    window_size: int
    overlap: int

@ dataclass(frozen=True)
class PreprocessingConfig:
    processed_data_path: Path
    signal_config: SignalProcessingConfig
    temp_baseline:float
    artifact_threshold: float