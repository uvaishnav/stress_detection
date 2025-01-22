import pytest
import numpy as np
import pandas as pd
from StressDetection.config.configuration import ConfigurationManager
from StressDetection.entity.entity import SignalProcessingConfig

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    n_samples = 1000
    time = np.linspace(0, 250, n_samples)
    
    data = pd.DataFrame({
        'BVP': np.sin(2 * np.pi * 1.2 * time),
        'EDA': 5 + 0.5 * np.sin(2 * np.pi * 0.1 * time),
        'TEMP': 36 + 0.5 * np.sin(2 * np.pi * 0.05 * time),
        'ACC_X': np.random.normal(0, 0.1, n_samples),
        'ACC_Y': np.random.normal(0, 0.1, n_samples),
        'ACC_Z': np.random.normal(1, 0.1, n_samples)
    })
    
    return data

@pytest.fixture
def signal_config():
    """Create test configuration."""
    return SignalProcessingConfig(
        common_sampling_rate=4,
        bvp_filter_low=0.7,
        bvp_filter_high=1.8,
        eda_filter_low=0.05,
        eda_filter_high=1.0,
        acc_threshold=0.8,
        motion_window=32,
        window_size=60,
        overlap=30,
        quality_threshold=0.85,  # Added default
        physiological_ranges={  # Added defaults
            'bvp_min': -1.0,
            'bvp_max': 1.0,
            'eda_min': 4.0,
            'eda_max': 6.0,
            'temp_min': 35.0,
            'temp_max': 37.0
        }
    )

