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
    jerking_threshold: float
    motion_window : int

    # window Parameters
    common_sampling_rate: int
    window_size: int
    overlap: int

@ dataclass(frozen=True)
class PreprocessingConfig:
    processed_data_path: Path
    temp_baseline:float
    artifact_threshold: float
    processed_data_target_path : Path


@dataclass
class FeatureExtractionConfig:
    sampling_rate: int
    hrv_features: bool
    time_domain_features: bool
    quality_threshold : float
    frequency_domain_features: bool
    motion_features: bool
    temperature_features: bool
    eda_features: bool
    target_path: Path