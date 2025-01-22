import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import pytest
import numpy as np
import pandas as pd
from StressDetection.components.signal_processing import SignalProcessor, SignalProcessingConfig


def test_end_to_end_pipeline(signal_config, large_sample_data):
    """Test complete signal processing pipeline."""
    # Create test config with more lenient thresholds
    test_config = SignalProcessingConfig(
        bvp_filter_low=signal_config.bvp_filter_low,
        bvp_filter_high=signal_config.bvp_filter_high,
        eda_filter_low=signal_config.eda_filter_low,
        eda_filter_high=signal_config.eda_filter_high,
        quality_threshold=0.2,  # Very lenient threshold
        acc_threshold=1.0,
        motion_window=32,        
        physiological_ranges={  # Match to generated signals
            'bvp_min': -1.0,
            'bvp_max': 1.0,
            'eda_min': 4.0,
            'eda_max': 6.0,
            'temp_min': 35.0,
            'temp_max': 37.0
        },
        common_sampling_rate=signal_config.common_sampling_rate,
        window_size=signal_config.window_size,
        overlap=signal_config.overlap
    )
    
    # Generate clean test data
    sample_rate = test_config.common_sampling_rate
    t = np.arange(len(large_sample_data)) / sample_rate
    
    # Clean physiological signals
    bvp = 0.6 * np.sin(2 * np.pi * 1.1 * t) 
    eda = 5.0 + 0.4 * np.sin(2 * np.pi * 0.1 * t)
    
    # First half - clean data
    half_point = len(large_sample_data) // 2
    large_sample_data.iloc[:half_point] = pd.DataFrame({
        'BVP': bvp[:half_point],
        'EDA': eda[:half_point],
        'TEMP': 36.0,
        'ACC_X': 0.0,
        'ACC_Y': 0.0,
        'ACC_Z': 1.0
    }, index=large_sample_data.index[:half_point])
    
    # Second half - mild noise
    noise_level = 0.1  # Reduced noise
    large_sample_data.iloc[half_point:] = pd.DataFrame({
        'BVP': bvp[half_point:] + noise_level * np.random.randn(len(large_sample_data) - half_point),
        'EDA': eda[half_point:] + noise_level * np.random.randn(len(large_sample_data) - half_point),
        'TEMP': 36.0,
        'ACC_X': 0.05 * np.random.randn(len(large_sample_data) - half_point),  # Less motion
        'ACC_Y': 0.05 * np.random.randn(len(large_sample_data) - half_point),
        'ACC_Z': 1.0
    }, index=large_sample_data.index[half_point:])
    
    # Process signals
    processor = SignalProcessor(test_config)
    segments, scores = processor.process_signals(large_sample_data)
    
    # Enhanced debug output
    print("\nTest Configuration:")
    print(f"Quality threshold: {test_config.quality_threshold}")
    print(f"Motion threshold: {test_config.acc_threshold}")
    print(f"BVP range: [{np.min(large_sample_data['BVP']):.2f}, {np.max(large_sample_data['BVP']):.2f}]")
    print(f"EDA range: [{np.min(large_sample_data['EDA']):.2f}, {np.max(large_sample_data['EDA']):.2f}]")
    print(f"ACC range: [{np.min(large_sample_data['ACC_X']):.2f}, {np.max(large_sample_data['ACC_X']):.2f}]")
    print("\nResults:")
    print(f"Segments: {len(segments)}")
    print(f"Score range: [{min(scores) if scores else 0:.3f}, {max(scores) if scores else 0:.3f}]")
    
    
    # Validate outputs
    assert len(segments) > 0, "Should produce valid segments"
    assert len(scores) == len(segments), "Scores should match segments"
    assert all(len(seg) == signal_config.window_size * signal_config.common_sampling_rate 
              for seg in segments), "Segments should have correct length"
    
    # Check quality thresholds
    assert all(score >= signal_config.quality_threshold for score in scores), \
        "All scores should meet threshold"



def calculate_max_segments(data_length, window_size, overlap):
    window_samples = window_size * 4  # Assuming 4Hz sampling rate
    overlap_samples = overlap * 4
    stride = window_samples - overlap_samples
    return (data_length - window_samples) // stride + 1

def test_motion_artifact_handling(strict_config, large_sample_data):
    """Test handling of motion artifacts in data."""
    artifact_period = slice(1000, 3000)
    large_sample_data.loc[artifact_period, ['ACC_X', 'ACC_Y', 'ACC_Z']] = 100.0

    processor = SignalProcessor(strict_config)
    segments, scores = processor.process_signals(large_sample_data)

    max_segments = calculate_max_segments(
        len(large_sample_data),
        strict_config.window_size,
        strict_config.overlap
    )


    print(f"Number of segments: {len(segments)}")
    print(f"Max segments: {max_segments}")
    print(f"Threshold: {max_segments * 0.3}")
    print(f"Scores: {scores}")

    assert len(segments) < max_segments * 0.3

def test_signal_quality_assessment(strict_config, large_sample_data):
    """Test signal quality assessment with controlled noise."""
    np.random.seed(42)


    sample_rate = strict_config.common_sampling_rate
    window_samples = strict_config.window_size * sample_rate
    t = np.arange(len(large_sample_data)) / sample_rate

    bvp_clean = 0.6 * np.sin(2 * np.pi * 1.1 * t)
    eda_clean = 5.0 + 0.4 * np.sin(2 * np.pi * 0.1 * t)

    clean_section = 8 * window_samples
    large_sample_data.iloc[:clean_section] = pd.DataFrame({
        'BVP': bvp_clean[:clean_section],
        'EDA': eda_clean[:clean_section],
        'TEMP': 36.0,
        'ACC_X': 0.0,
        'ACC_Y': 0.0,
        'ACC_Z': 1.0
    }, index=large_sample_data.index[:clean_section])

    mild_noise = 10 * window_samples
    noise_level = 0.1
    large_sample_data.iloc[clean_section:mild_noise] = pd.DataFrame({
        'BVP': bvp_clean[clean_section:mild_noise] + np.random.normal(0, noise_level * 0.6, mild_noise-clean_section),
        'EDA': eda_clean[clean_section:mild_noise] + np.random.normal(0, noise_level * 0.4, mild_noise-clean_section),
        'TEMP': 36.0,
        'ACC_X': 0.2 * np.random.randn(mild_noise-clean_section),
        'ACC_Y': 0.2 * np.random.randn(mild_noise-clean_section),
        'ACC_Z': 1.0
    }, index=large_sample_data.index[clean_section:mild_noise])

    for i in range(10, len(large_sample_data) // window_samples):
        start = i * window_samples
        end = min((i + 1) * window_samples, len(large_sample_data))
        samples = end - start
        noise_factor = min(0.15 * (i-9), 0.8)

        motion = 0.3 * np.random.randn(samples)
        large_sample_data.iloc[start:end] = pd.DataFrame({
            'BVP': bvp_clean[start:end] + noise_factor * motion,
            'EDA': eda_clean[start:end] + noise_factor * 0.5 * motion,
            'TEMP': 36.0,
            'ACC_X': motion,
            'ACC_Y': 0.3 * np.random.randn(samples),
            'ACC_Z': 1.0 + 0.3 * motion
        }, index=large_sample_data.index[start:end])

    processor = SignalProcessor(strict_config)
    segments, scores = processor.process_signals(large_sample_data)
    max_segments = calculate_max_segments(len(large_sample_data), strict_config.window_size, strict_config.overlap)
    print("\nTest Configuration:")
    print(f"Quality threshold: {strict_config.quality_threshold}")
    print(f"BVP signal: {np.ptp(bvp_clean[:window_samples])/2:.2f}V @ 1.1Hz")
    print(f"EDA signal: {np.ptp(eda_clean[:window_samples])/2:.2f}V @ 0.1Hz")
    print(f"Max noise level: {noise_factor*100:.0f}%")
    print("\nResults:")
    print(f"Segments: {len(segments)}/{max_segments}")
    print(f"Score range: [{np.min(scores):.3f}, {np.max(scores):.3f}]")

    assert len(segments) > 0, "Should retain valid segments"
    assert len(segments) < max_segments * 0.8, "Should filter noisy segments"

def test_kalman_filter(noisy_signal):
    """Test Kalman filter performance on a noisy signal."""
    processor = SignalProcessor(SignalProcessingConfig(
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
    ))

    filtered_signal = processor.kalman_filter(noisy_signal)
    
    # Assert that the filtered signal has less noise
    assert np.std(filtered_signal) < np.std(noisy_signal), "Kalman filter did not reduce noise"

    # Plot the results for visual inspection (optional)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(noisy_signal, label='Noisy Signal')
    plt.plot(filtered_signal, label='Filtered Signal', linewidth=2)
    plt.legend()
    plt.title('Kalman Filter Performance')
    plt.show()