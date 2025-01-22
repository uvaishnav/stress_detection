import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt
from typing import Tuple, Dict, List
import pandas as pd

from StressDetection.entity.entity import SignalProcessingConfig

class SignalProcessor:
    def __init__(self, config: SignalProcessingConfig):
        self.config = config

    def design_filters(self) -> Dict:
        """Design bandpass filters for BVP and EDA signals."""
        fs = self.config.common_sampling_rate  # Sampling rate (4Hz)
        nyquist = fs / 2

        n = fs - 1  # Filter order

        # BVP bandpass filter
        bvp_b, bvp_a = butter(
            N=n,
            Wn=[min(self.config.bvp_filter_low, nyquist - 0.1),
                min(self.config.bvp_filter_high, nyquist - 0.1)],
            btype='bandpass',
            fs=fs
        )

        # EDA bandpass filter
        eda_b, eda_a = butter(
            N=n,
            Wn=min(self.config.eda_filter_high, nyquist - 0.1),
            btype='low',
            fs=fs
        )

        return {
            'bvp': (bvp_b, bvp_a),
            'eda': (eda_b, eda_a)
        }
    
    def kalman_filter(self, signal: np.ndarray) -> np.ndarray:
        """Apply Kalman filtering to the signal."""
        n = len(signal)
        Q = 1e-5  # Process variance
        R = 0.1**2  # Estimate of measurement variance, change to see effect

        xhat = np.zeros(n)  # a posteri estimate of x
        P = np.zeros(n)  # a posteri error estimate
        xhatminus = np.zeros(n)  # a priori estimate of x
        Pminus = np.zeros(n)  # a priori error estimate
        K = np.zeros(n)  # gain or blending factor

        # initial guesses
        xhat[0] = signal[0]
        P[0] = 1.0

        for k in range(1, n):
            # time update
            xhatminus[k] = xhat[k-1]
            Pminus[k] = P[k-1] + Q

            # measurement update
            K[k] = Pminus[k] / (Pminus[k] + R)
            xhat[k] = xhatminus[k] + K[k] * (signal[k] - xhatminus[k])
            P[k] = (1 - K[k]) * Pminus[k]

        return xhat

    def detect_motion(self, acc_data: np.ndarray) -> np.ndarray:
        """Enhanced motion detection with magnitude and jerk analysis."""
        acc_centered = acc_data - np.array([0, 0, 1.0])  # Remove gravity
        acc_magnitude = np.sqrt(np.sum(acc_centered ** 2, axis=1))
        acc_diff = np.diff(acc_magnitude, prepend=acc_magnitude[0])
        jerk = np.abs(acc_diff)

        motion_mask = (
            (acc_magnitude > self.config.acc_threshold) |
            (jerk > self.config.acc_threshold)
        )

        window = np.ones(self.config.motion_window)
        smoothed = np.convolve(motion_mask.astype(float), window / len(window), mode='same')

        # Debugging statements
        print(f"acc_magnitude: {acc_magnitude}")
        print(f"jerk: {jerk}")
        print(f"motion_mask: {motion_mask}")
        print(f"smoothed: {smoothed}")

        return smoothed > 0.5

    def apply_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply designed filters to physiological signals."""
        filters = self.design_filters()
        filtered_data = data.copy()

        filtered_data['BVP'] = filtfilt(
            filters['bvp'][0],
            filters['bvp'][1],
            data['BVP']
        )

        filtered_data['EDA'] = filtfilt(
            filters['eda'][0],
            filters['eda'][1],
            data['EDA']
        )

        # Apply Kalman filter to smooth the signals
        filtered_data['BVP'] = self.kalman_filter(filtered_data['BVP'])
        filtered_data['EDA'] = self.kalman_filter(filtered_data['EDA'])

        return filtered_data

    def assess_signal_quality(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate signal quality with adaptive thresholds."""
        motion_mask = ~self.detect_motion(data[['ACC_X', 'ACC_Y', 'ACC_Z']].values)
        ranges = self.config.physiological_ranges

        bvp_valid = (ranges['bvp_min'] <= data['BVP']) & (data['BVP'] <= ranges['bvp_max'])
        eda_valid = (ranges['eda_min'] <= data['EDA']) & (data['EDA'] <= ranges['eda_max'])
        temp_valid = (ranges['temp_min'] <= data['TEMP']) & (data['TEMP'] <= ranges['temp_max'])

        short_window = 8
        long_window = 24

        def get_zero_crossings(signal):
            diff_signal = np.diff(signal)
            crossings = np.zeros(len(signal), dtype=bool)
            sign_changes = np.signbit(diff_signal[:-1]) != np.signbit(diff_signal[1:])
            crossings[1:-1] = sign_changes
            return crossings

        bvp_zero_crossings = get_zero_crossings(data['BVP'].values)
        eda_zero_crossings = get_zero_crossings(data['EDA'].values - data['EDA'].mean())

        bvp_std_short = data['BVP'].rolling(window=short_window, center=True).std().fillna(0)
        eda_std_short = data['EDA'].rolling(window=short_window, center=True).std().fillna(0)
        bvp_std_long = data['BVP'].rolling(window=long_window, center=True).std().fillna(0)
        eda_std_long = data['EDA'].rolling(window=long_window, center=True).std().fillna(0)

        bvp_stable = (bvp_std_short < 0.15) & (bvp_std_long < 0.2) & bvp_zero_crossings
        eda_stable = (eda_std_short < 0.1) & (eda_std_long < 0.15) & ~eda_zero_crossings

        signal_stable = (bvp_stable & eda_stable) | self._check_frequency_stability(data['BVP'])
        stability_score = np.mean(signal_stable)

        perfect_quality = motion_mask & bvp_valid & eda_valid & signal_stable
        high_quality = motion_mask & bvp_valid & eda_valid & (bvp_stable | eda_stable)
        base_quality = motion_mask & bvp_valid & eda_valid

        quality_mask = np.where(perfect_quality,
                                0.98 + 0.02 * stability_score,
                                np.where(high_quality,
                                         0.95 + 0.03 * stability_score,
                                         np.where(base_quality,
                                                  0.90 + 0.05 * stability_score,
                                                  0.20)))

        return quality_mask

    def _check_frequency_stability(self, signal: pd.Series) -> bool:
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        if len(zero_crossings) < 2:
            return False

        intervals = np.diff(zero_crossings)
        cv = np.std(intervals) / np.mean(intervals)
        return cv < 0.5

    def segment_signals(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        window_samples = self.config.window_size * self.config.common_sampling_rate
        overlap_samples = self.config.overlap * self.config.common_sampling_rate
        stride = window_samples - overlap_samples

        segments = []
        for start in range(0, len(data) - window_samples + 1, stride):
            segment = data.iloc[start:start + window_samples]
            segments.append(segment)

        return segments

    def calculate_segment_quality(self, quality_mask: np.ndarray) -> List[float]:
        window_samples = self.config.window_size * self.config.common_sampling_rate
        overlap_samples = self.config.overlap * self.config.common_sampling_rate
        stride = window_samples - overlap_samples

        quality_scores = []
        for start in range(0, len(quality_mask) - window_samples + 1, stride):
            segment = quality_mask[start:start + window_samples].astype(float)
            score = float(np.mean(segment))
            quality_scores.append(score)

        return quality_scores

    def process_signals(self, data: pd.DataFrame) -> Tuple[List[pd.DataFrame], List[float]]:
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
        return {
            'total_segments': len(segments),
            'total_duration': len(segments) * self.config.window_size,
            'valid_duration_ratio': len(segments) * self.config.window_size / (len(segments) * 60)
        }