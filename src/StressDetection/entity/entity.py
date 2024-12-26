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


