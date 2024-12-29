import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt
from typing import Tuple, Dict, List
import pandas as pd

from StressDetection.entity.entity import SignalProcessingConfig

class SignalProcessor:
    def __init__(self, config:SignalProcessingConfig):
        self.config = config

    def design_filters(self)->Dict:
        """Design bandpass filters for BVP and EDA signals."""

        fs = self.config.common_sampling_rate  # Sampling rate (4Hz)
        nyquist = fs/2

        n = self.config.common_sampling_rate-1  # Filter order
        # Lower filter order instead of using sampling rate

        # BVP bandpass filter
        # Scale frequencies since they can't exceed fs/2 (Nyquist frequency ) (2Hz)
        bvp_b, bvp_a = butter(
            N = n, 
            Wn=[min(self.config.bvp_filter_low, nyquist-0.1), 
                min(self.config.bvp_filter_high, nyquist-0.1)],
            btype = 'bandpass',
            fs=fs
        ) 
        # Why Bandpass?
        # Keeps frequencies between 0.7-3.7Hz
        # Heart rate range: 42-222 BPM
        # Removes noise and baseline drift

        # EDA bandpass filter
        eda_b, eda_a = butter(
            N = n,
            Wn=min(self.config.eda_filter_high, nyquist-0.1),
            btype = 'low',
            fs=fs # # Use sampling frequency parameter
        )
        # Why Lowpass?
        # Keeps frequencies below 0.4Hz
        # EDA signal is slow varying
        # Removes high frequency noise


        return {
            'bvp': (bvp_b, bvp_a),
            'eda': (eda_b, eda_a)
        }

    def detect_motion(self, acc_data: np.ndarray) -> np.ndarray:
        """Enhanced motion detection with magnitude and jerk analysis."""
        # Calculate acceleration magnitude
        acc_centered = acc_data - np.array([0, 0, 1.0])  # Remove gravity
        acc_magnitude = np.sqrt(np.sum(acc_centered**2, axis=1))
        
        # Calculate jerk (rate of change)
        acc_diff = np.diff(acc_magnitude, prepend=acc_magnitude[0])
        jerk = np.abs(acc_diff)
        
        # More stringent thresholds for motion detection
        motion_mask = (
            (acc_magnitude > self.config.acc_threshold * 1.5) |  # Increased magnitude threshold
            (jerk > self.config.acc_threshold * 1.2)  # Increased jerk threshold
        )
        
        # Use shorter window and higher threshold for smoothing
        window = np.ones(min(6, self.config.motion_window))  # Reduced window size
        smoothed = np.convolve(motion_mask.astype(float), window/len(window), mode='same')
        return smoothed > 0.6  # Increased threshold for motion detection   
 
    def apply_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply designed filters to physiological signals."""
        filters = self.design_filters()
        filtered_data = data.copy()
        
        # Apply BVP filter
        filtered_data['BVP'] = filtfilt(
            filters['bvp'][0], 
            filters['bvp'][1], 
            data['BVP']
        )
        
        # Apply EDA filter
        filtered_data['EDA'] = filtfilt(
            filters['eda'][0], 
            filters['eda'][1], 
            data['EDA']
        )
        
        return filtered_data

    def assess_signal_quality(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate signal quality with tier-based assessment."""
        # Basic checks
        motion_mask = ~self.detect_motion(data[['ACC_X', 'ACC_Y', 'ACC_Z']].values)
        ranges = self.config.physiological_ranges
        
        # Validity checks
        bvp_valid = (ranges['bvp_min'] <= data['BVP']) & (data['BVP'] <= ranges['bvp_max'])
        eda_valid = (ranges['eda_min'] <= data['EDA']) & (data['EDA'] <= ranges['eda_max'])
        temp_valid = (ranges['temp_min'] <= data['TEMP']) & (data['TEMP'] <= ranges['temp_max'])
        
        # Enhanced stability analysis with relaxed thresholds
        short_window = 8   # Slightly larger for smoother analysis
        long_window = 24   # Longer trend window
        
        # Calculate signal metrics
        bvp_diff = np.abs(np.diff(data['BVP'], prepend=data['BVP'].iloc[0]))
        eda_diff = np.abs(np.diff(data['EDA'], prepend=data['EDA'].iloc[0]))
        
        # Rolling statistics
        bvp_std_short = pd.Series(data['BVP']).rolling(window=short_window).std().bfill()
        eda_std_short = pd.Series(data['EDA']).rolling(window=short_window).std().bfill()
        bvp_std_long = pd.Series(data['BVP']).rolling(window=long_window).std().bfill()
        eda_std_long = pd.Series(data['EDA']).rolling(window=long_window).std().bfill()
        
        # Relaxed stability criteria
        bvp_stable = (bvp_diff < 0.15) & (bvp_std_short < 0.2) & (bvp_std_long < 0.25)
        eda_stable = (eda_diff < 0.08) & (eda_std_short < 0.1) & (eda_std_long < 0.15)
        
        # Signal variance check (new)
        bvp_var_ok = pd.Series(data['BVP']).rolling(window=long_window).var().bfill() < 0.3
        eda_var_ok = pd.Series(data['EDA']).rolling(window=long_window).var().bfill() < 0.2
        
        # Combined stability with variance
        signal_stable = (bvp_stable & eda_stable) | (bvp_var_ok & eda_var_ok)
        stability_score = np.mean(signal_stable)
        
        # Debug output
        print("\nQuality Mask Details:")
        print(f"Motion valid: {np.mean(motion_mask):.2f}")
        print(f"BVP valid: {np.mean(bvp_valid):.2f}")
        print(f"EDA valid: {np.mean(eda_valid):.2f}")
        print(f"Signal stability: {stability_score:.2f}")
        
        # Quality tiers with higher base scores
        perfect_quality = motion_mask & bvp_valid & eda_valid & signal_stable
        high_quality = motion_mask & bvp_valid & eda_valid & (bvp_stable | eda_stable)
        base_quality = motion_mask & bvp_valid & eda_valid
        
        # Adjusted tier scoring
        quality_mask = np.where(perfect_quality, 
                            0.98 + 0.02 * stability_score,    # Tier 1: 0.98-1.0
                            np.where(high_quality,
                                    0.95 + 0.03 * stability_score,  # Tier 2: 0.95-0.98
                                    np.where(base_quality,
                                            0.90 + 0.05 * stability_score,  # Tier 3: 0.90-0.95
                                            0.20 + 0.10 * stability_score))) # Poor: 0.20-0.30
        
        print(f"Final quality: {np.mean(quality_mask):.2f}")
        return quality_mask

    def _check_frequency_stability(self, signal: pd.Series) -> bool:
        """Check if signal maintains expected frequency characteristics."""
        # Simple frequency check using zero crossings
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        if len(zero_crossings) < 2:
            return False
            
        # Check if intervals between zero crossings are relatively stable
        intervals = np.diff(zero_crossings)
        cv = np.std(intervals) / np.mean(intervals)
        return cv < 0.5  # Coefficient of variation threshold
    
    def segment_signals(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """Segment signals into windows with overlap."""
        window_samples = self.config.window_size * self.config.common_sampling_rate  # 4Hz sampling rate
        overlap_samples = self.config.overlap * self.config.common_sampling_rate
        stride = window_samples - overlap_samples
        
        segments = []
        for start in range(0, len(data) - window_samples + 1, stride):
            segment = data.iloc[start:start + window_samples]
            segments.append(segment)
        
        return segments
    
    def calculate_segment_quality(self, quality_mask: np.ndarray) -> List[float]:
        """Calculate quality scores for segments from quality mask."""
        window_samples = self.config.window_size * self.config.common_sampling_rate
        overlap_samples = self.config.overlap * self.config.common_sampling_rate
        stride = window_samples - overlap_samples
        
        quality_scores = []
        for start in range(0, len(quality_mask) - window_samples + 1, stride):
            # Convert to numpy array and ensure float type
            segment = quality_mask[start:start + window_samples].astype(float)
            
            # Simple mean quality score
            score = float(np.mean(segment))
            quality_scores.append(score)
        
        return quality_scores
    
    def process_signals(self, data: pd.DataFrame) -> Tuple[List[pd.DataFrame], List[float]]:
        """Process physiological signals and assess quality.
        
        Args:
            data: DataFrame containing physiological signals (BVP, EDA, TEMP, ACC)
            
        Returns:
            Tuple containing:
                - List of valid signal segments as DataFrames
                - List of quality scores for each segment
                
        Note:
            Segments are considered valid if their quality score exceeds
            the configured quality threshold.
        """
        filtered_data = self.apply_filters(data)
        quality_mask = self.assess_signal_quality(filtered_data)
        segments = self.segment_signals(filtered_data)
        quality_scores = self.calculate_segment_quality(quality_mask)
        
        valid_segments = [
            segment for segment, quality in zip(segments, quality_scores)
            if quality >= self.config.quality_threshold
        ]
        
        return valid_segments, quality_scores
    
    def get_segment_statistics(self, segments: List[pd.DataFrame]) -> Dict:
        """Calculate statistics about segmentation."""
        return {
            'total_segments': len(segments),
            'total_duration': len(segments) * self.config.window_size,
            'valid_duration_ratio': len(segments) * self.config.window_size / (len(segments) * 60)
        }
    


    