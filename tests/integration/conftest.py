import pytest
import numpy as np
import pandas as pd
from dataclasses import replace
from StressDetection.entity.entity import SignalProcessingConfig


@pytest.fixture(scope="function")
def strict_config(signal_config):
    """Create stricter configuration for testing."""
    return replace(
        signal_config,
        acc_threshold=0.5,  # Lower threshold for motion
        quality_threshold=0.8,  # Higher quality requirement
    )

@pytest.fixture
def large_sample_data():
    """Generate large test dataset."""
    n_samples = 10000
    time = np.linspace(0, 2500, n_samples)
    
    return pd.DataFrame({
        'BVP': np.sin(2 * np.pi * 1.2 * time),
        'EDA': 5 + 0.5 * np.sin(2 * np.pi * 0.1 * time),
        'TEMP': 36 + 0.5 * np.sin(2 * np.pi * 0.05 * time),
        'ACC_X': np.random.normal(0, 0.1, n_samples),
        'ACC_Y': np.random.normal(0, 0.1, n_samples),
        'ACC_Z': np.random.normal(1, 0.1, n_samples)
    })

@pytest.fixture
def signal_config():
    """Create test configuration."""
    return SignalProcessingConfig(
        common_sampling_rate=4,
        bvp_filter_low=0.7,
        bvp_filter_high=1.8,
        eda_filter_low=0.05,
        eda_filter_high=0.8,
        acc_threshold=1.2,
        motion_window=8,
        window_size=60,
        overlap=30,
        quality_threshold=0.7,
        physiological_ranges={
            'bvp_min': -10.0,
            'bvp_max': 10.0,
            'eda_min': 0.0,
            'eda_max': 25.0,
            'temp_min': 30.0,
            'temp_max': 40.0
        }
    )