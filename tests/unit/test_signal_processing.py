import pytest
import numpy as np
from StressDetection.components.signal_processing import SignalProcessor

@pytest.mark.signal
def test_filter_design(signal_config):
    """Test filter creation and parameters.
    
    Args:
        signal_config: Fixture providing signal configuration
    """
    processor = SignalProcessor(signal_config)
    filters = processor.design_filters()
    
    assert isinstance(filters, dict), "Filters should be returned as dictionary"
    assert all(key in filters for key in ['bvp', 'eda']), "Missing required filters"
    assert all(len(coef) == 2 for coef in filters.values()), "Each filter should have b,a coefficients"

@pytest.mark.signal
def test_motion_detection(signal_config, sample_data):
    processor = SignalProcessor(signal_config)
    
    # Create clearer motion artifact
    motion_period = slice(500, 550)
    sample_data.loc[motion_period, ['ACC_X', 'ACC_Y', 'ACC_Z']] = 5.0  # Increase magnitude
    
    acc_data = sample_data[['ACC_X', 'ACC_Y', 'ACC_Z']].values
    motion_mask = processor.detect_motion(acc_data)
    
    # Check specific points
    assert motion_mask[525], "Should detect motion in artificial motion period"
    assert not motion_mask[100], "Should not detect motion in clean period"

@pytest.mark.signal
def test_signal_quality(signal_config, sample_data):
    """Test signal quality assessment.
    
    Args:
        signal_config: Fixture providing signal configuration
        sample_data: Fixture providing sample physiological data
    """
    processor = SignalProcessor(signal_config)
    quality_mask = processor.assess_signal_quality(sample_data)
    
    quality_threshold = 0.9
    mean_quality = np.mean(quality_mask)
    assert mean_quality > quality_threshold, f"Quality below threshold: {mean_quality:.2f}"

@pytest.mark.signal
def test_segmentation(signal_config, sample_data):
    """Test signal segmentation.
    
    Args:
        signal_config: Fixture providing signal configuration
        sample_data: Fixture providing sample physiological data
    """
    processor = SignalProcessor(signal_config)
    segments = processor.segment_signals(sample_data)
    
    sampling_rate = 4  # Hz
    expected_samples = signal_config.window_size * sampling_rate
    
    assert segments, "No segments were generated"
    assert all(len(seg) == expected_samples for seg in segments), "Incorrect segment length"

@pytest.mark.signal
def test_end_to_end_processing(signal_config, sample_data):
    """Test complete signal processing pipeline.
    
    Args:
        signal_config: Fixture providing signal configuration
        sample_data: Fixture providing sample physiological data
    """
    processor = SignalProcessor(signal_config)
    valid_segments, quality_scores = processor.process_signals(sample_data)
    
    assert valid_segments, "No valid segments produced"
    assert len(valid_segments) == len(quality_scores), "Segments and scores count mismatch"
    assert all(0 <= score <= 1 for score in quality_scores), "Quality scores out of range [0,1]"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])