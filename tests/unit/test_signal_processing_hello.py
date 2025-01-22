import pytest
import pandas as pd
import numpy as np
from src.StressDetection.components.signal_processing import SignalProcessor, SignalProcessingConfig

@pytest.fixture
def signal_config():
    return SignalProcessingConfig(
        bvp_filter_low=0.7,
        bvp_filter_high=3.7,
        eda_filter_low=0.05,
        eda_filter_high=1.0,
        physiological_ranges={
            'bvp_min': 0.0,
            'bvp_max': 1.0,
            'eda_min': 0.0,
            'eda_max': 1.0,
            'temp_min': 30.0,
            'temp_max': 40.0
        },
        acc_threshold=0.8,
        motion_window=32,
        common_sampling_rate=4,
        window_size=60,
        overlap=30
    )

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'BVP': np.random.uniform(0.0, 1.0, 1000),
        'EDA': np.random.uniform(0.0, 1.0, 1000),
        'TEMP': np.random.uniform(30.0, 40.0, 1000),
        'ACC_X': np.random.normal(0, 0.1, 1000),
        'ACC_Y': np.random.normal(0, 0.1, 1000),
        'ACC_Z': np.random.normal(1, 0.1, 1000)
    })

def test_assess_signal_quality_valid(signal_config, sample_data):
    processor = SignalProcessor(signal_config)
    quality_mask = processor.assess_signal_quality(sample_data)
    assert quality_mask.shape == (1000,)

def test_assess_signal_quality_edge_cases(signal_config):
    processor = SignalProcessor(signal_config)
    edge_data = pd.DataFrame({
        'BVP': [0.0, 1.0],
        'EDA': [0.0, 1.0],
        'TEMP': [30.0, 40.0],
        'ACC_X': [0.0, 0.0],
        'ACC_Y': [0.0, 0.0],
        'ACC_Z': [1.0, 1.0]
    })
    quality_mask = processor.assess_signal_quality(edge_data)
    assert quality_mask.shape == (2,)

def test_assess_signal_quality_invalid_data(signal_config):
    processor = SignalProcessor(signal_config)
    invalid_data = pd.DataFrame({
        'BVP': [1.5, 2.0],
        'EDA': [1.5, 2.0],
        'TEMP': [50.0, 60.0],
        'ACC_X': [0.0, 0.0],
        'ACC_Y': [0.0, 0.0],
        'ACC_Z': [1.0, 1.0]
    })
    with pytest.raises(ValueError):
        processor.assess_signal_quality(invalid_data)