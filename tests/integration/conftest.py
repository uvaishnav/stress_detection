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
        acc_threshold=0.1,  # Lower threshold for motion
        quality_threshold=0.85,  # Higher quality requirement
        bvp_filter_low=0.7,
        bvp_filter_high=1.8,
        eda_filter_low=0.05,
        eda_filter_high=0.8,
        motion_window=32,
        physiological_ranges={
            'bvp_min': -1.0,
            'bvp_max': 1.0,
            'eda_min': 4.0,
            'eda_max': 6.0,
            'temp_min': 35.0,
            'temp_max': 37.0
        }
    )

@pytest.fixture
def large_sample_data():
    """Generate large test dataset."""
    np.random.seed(42)
    sample_size = 10000
    return pd.DataFrame({
        'BVP': np.random.uniform(-1.0, 1.0, sample_size),
        'EDA': np.random.uniform(4.0, 6.0, sample_size),
        'TEMP': np.random.uniform(35.0, 37.0, sample_size),
        'ACC_X': np.random.normal(0, 0.1, sample_size),
        'ACC_Y': np.random.normal(0, 0.1, sample_size),
        'ACC_Z': np.random.normal(1, 0.1, sample_size)
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
        acc_threshold=0.1,
        motion_window=32,
        window_size=60,
        overlap=30,
        quality_threshold=0.85,
        physiological_ranges={
            'bvp_min': -1.0,
            'bvp_max': 1.0,
            'eda_min': 4.0,
            'eda_max': 6.0,
            'temp_min': 35.0,
            'temp_max': 37.0
        }
    )

@pytest.fixture
def noisy_signal():
    """Generate a noisy signal for testing."""
    np.random.seed(42)
    n_samples = 1000
    signal = np.sin(np.linspace(0, 10 * np.pi, n_samples))  # Sine wave
    noise = np.random.normal(0, 0.5, n_samples)  # Gaussian noise
    noisy_signal = signal + noise
    return noisy_signal