import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import resample
from StressDetection.entity.entity import FeatureExtractionConfig

class FeatureExtractor:
    def __init__(self, config: FeatureExtractionConfig):
        self.config = config

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

    def preprocess_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the raw signals."""
        # Resample all signals to 64Hz
        target_sampling_rate = 64
        original_sampling_rate = 4

        bvp_resampled = resample(data['BVP'], int(len(data['BVP']) * target_sampling_rate / original_sampling_rate))
        acc_x_resampled = resample(data['ACC_X'], int(len(data['ACC_X']) * target_sampling_rate / original_sampling_rate))
        acc_y_resampled = resample(data['ACC_Y'], int(len(data['ACC_Y']) * target_sampling_rate / original_sampling_rate))
        acc_z_resampled = resample(data['ACC_Z'], int(len(data['ACC_Z']) * target_sampling_rate / original_sampling_rate))
        temp_resampled = resample(data['TEMP'], int(len(data['TEMP']) * target_sampling_rate / original_sampling_rate))
        eda_resampled = resample(data['EDA'], int(len(data['EDA']) * target_sampling_rate / original_sampling_rate))

        # Create new DataFrame for resampled signals
        resampled_data = pd.DataFrame({
            'BVP': bvp_resampled,
            'ACC_X': acc_x_resampled,
            'ACC_Y': acc_y_resampled,
            'ACC_Z': acc_z_resampled,
            'TEMP': temp_resampled,
            'EDA': eda_resampled
        })

        # Clean BVP signal
        resampled_data['BVP'] = nk.ppg_clean(resampled_data['BVP'], sampling_rate=target_sampling_rate)
        resampled_data['BVP'] = nk.signal_fillmissing(resampled_data['BVP'], method="linear")  # Fill missing values
        
        # Apply Kalman filter to ACC signals
        resampled_data['ACC_X'] = self.kalman_filter(resampled_data['ACC_X'])
        resampled_data['ACC_Y'] = self.kalman_filter(resampled_data['ACC_Y'])
        resampled_data['ACC_Z'] = self.kalman_filter(resampled_data['ACC_Z'])
        
        # Smooth TEMP signal
        resampled_data['TEMP'] = nk.signal_smooth(resampled_data['TEMP'], method='moving_average', size=5)
        
        # Clean EDA signal
        resampled_data['EDA'] = nk.eda_clean(resampled_data['EDA'], sampling_rate=target_sampling_rate)

        return resampled_data

    def extract_time_domain_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract time-domain features."""
        features = pd.DataFrame()
        features['BVP_mean'] = [data['BVP'].mean()]
        features['BVP_std'] = [data['BVP'].std()]
        features['ACC_X_mean'] = [data['ACC_X'].mean()]
        features['ACC_Y_mean'] = [data['ACC_Y'].mean()]
        features['ACC_Z_mean'] = [data['ACC_Z'].mean()]
        features['TEMP_mean'] = [data['TEMP'].mean()]
        features['EDA_mean'] = [data['EDA'].mean()]
        features['EDA_std'] = [data['EDA'].std()]
        return features

    def extract_frequency_domain_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract frequency-domain features."""
        features = pd.DataFrame()
        # Example: Power Spectral Density (PSD) for BVP
        psd = nk.signal_psd(data['BVP'], sampling_rate=64)
        features['BVP_psd_mean'] = [psd['Power'].mean()]
        features['BVP_psd_std'] = [psd['Power'].std()]
        return features

    def extract_motion_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract motion features from ACC."""
        features = pd.DataFrame()
        acc_magnitude = np.sqrt(data['ACC_X']**2 + data['ACC_Y']**2 + data['ACC_Z']**2)
        features['ACC_magnitude_mean'] = [acc_magnitude.mean()]
        features['ACC_magnitude_std'] = [acc_magnitude.std()]
        return features

    def extract_hrv_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract HRV features from BVP."""
        features = pd.DataFrame()
        signals, info = nk.ppg_process(data['BVP'], sampling_rate=64)
        hrv_indices = nk.hrv_time(signals, sampling_rate=64)
        for key, value in hrv_indices.items():
            features[key] = [value[0] if isinstance(value, pd.Series) else value]  # Ensure single value
        return features

    def extract_eda_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract EDA features."""
        features = pd.DataFrame()
        eda_signals, info = nk.eda_process(data['EDA'], sampling_rate=64)
        features['EDA_SCL_mean'] = [eda_signals['EDA_Tonic'].mean()]
        features['EDA_SCR_mean'] = [eda_signals['EDA_Phasic'].mean()]
        features['EDA_SCR_peaks'] = [len(info['SCR_Peaks'])]
        features['EDA_SCR_amplitude_mean'] = [eda_signals['SCR_Amplitude'].mean()]
        return features

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract all features from the data."""
        data = self.preprocess_signals(data)
        features = pd.DataFrame()

        if self.config.time_domain_features:
            time_features = self.extract_time_domain_features(data)
            features = pd.concat([features, time_features], axis=1)

        if self.config.frequency_domain_features:
            freq_features = self.extract_frequency_domain_features(data)
            features = pd.concat([features, freq_features], axis=1)

        if self.config.motion_features:
            motion_features = self.extract_motion_features(data)
            features = pd.concat([features, motion_features], axis=1)

        if self.config.hrv_features:
            hrv_features = self.extract_hrv_features(data)
            features = pd.concat([features, hrv_features], axis=1)

        if 'EDA' in data.columns and self.config.eda_features:
            eda_features = self.extract_eda_features(data)
            features = pd.concat([features, eda_features], axis=1)

        return features