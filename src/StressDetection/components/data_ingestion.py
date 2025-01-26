import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict, Tuple

from StressDetection import logger
from StressDetection.entity.entity import WesadDataIngestionConfig

class WESADLoader:
    """
    A class to load and process WESAD (Wearable Stress and Affect Detection) dataset.
    
    This class handles loading, validation, processing, and saving of physiological data
    from wearable sensors along with stress labels.
    """

    def __init__(self, config: WesadDataIngestionConfig):
        """
        Initialize the WESADLoader with configuration parameters.

        Args:
            config (WesadDataIngestionConfig): Configuration object containing data paths,
                                             sampling rates, and other parameters.
        """
        self.config = config

    def load_subjects(self, subject_id: int) -> Dict:
        """
        Load data for a specific subject from pickle file.

        Args:
            subject_id (int): ID of the subject to load.

        Returns:
            Dict: Dictionary containing subject's sensor data and labels.

        Raises:
            ValueError: If subject_id is not in valid_subjects.
            FileNotFoundError: If subject's data file is not found.
            pickle.UnpicklingError: If there's an error reading the pickle file.
        """
        if subject_id not in self.config.valid_subjects:
            raise ValueError(f"Invalid subject ID: {subject_id}")
        
        data_source_path = Path(self.config.data_source_path)
        file_path = data_source_path / f"S{subject_id}/S{subject_id}.pkl"
        
        try:
            with open(file_path, 'rb') as file:
                data = pickle.load(file, encoding='latin1')
            logger.info(f"Data loaded successfully for subject {subject_id}")
            logger.info(f"Data keys: {data.keys()}")
            return data
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            logger.error(f"Error loading data for subject {subject_id}: {str(e)}")
            raise e

    def validate_wrist_data(self, data: Dict) -> bool:
        """
        Validate the wrist sensor data for required signals and sampling rates.

        Args:
            data (Dict): Dictionary containing wrist sensor data.

        Returns:
            bool: True if data meets all validation criteria, False otherwise.
        """
        try:
            wrist_data = data['signal']['wrist']
            
            # Check for required signals
            if not all(signal in wrist_data for signal in self.config.required_signals):
                logger.warning("Missing required signals in wrist data")
                return False
            
            # Validate signal lengths
            label_length = len(data['label'])
            for signal, rate in self.config.sampling_rates['wrist'].items():
                logger.info(f"Signal: {signal}, Rate: {rate}")
                rate = float(rate)  # Ensure rate is numeric
                expected_length = label_length * (rate / 700)  # 700Hz is label rate
                actual_length = len(wrist_data[signal])
                
                if abs(expected_length - actual_length) > rate:
                    logger.warning(f"Signal length mismatch for {signal}")
                    return False
            
            return True
        except KeyError as e:
            logger.error(f"Data structure error: {str(e)}")
            return False

    def extract_wrist_data(self, subject_id: int) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Extract and process wrist sensor data and stress labels.
        
        Args:
            subject_id (int): Subject identifier
            
        Returns:
            Tuple[pd.DataFrame, np.ndarray]: Processed sensor data and binary labels
            
        Raises:
            ValueError: If data validation fails or there's a length mismatch
        """
        try:
            data = self.load_subjects(subject_id)
            if not self.validate_wrist_data(data):
                raise ValueError(f"Invalid wrist data for subject {subject_id}")
            
            # Normalize signal lengths
            normalized_data = self.normalize_signal_lengths(data)

            logger.info(f"Normalized data keys: {normalized_data.keys()}")
            logger.info(f"Normalized data lengths for BVP: {len(normalized_data['BVP'])}")
            logger.info(f"Normalized data lengths for ACC_X: {len(normalized_data['ACC_X'])}")
            logger.info(f"Normalized data lengths for ACC_Y: {len(normalized_data['ACC_Y'])}")
            logger.info(f"Normalized data lengths for ACC_Z: {len(normalized_data['ACC_Z'])}")
            logger.info(f"Normalized data lengths for TEMP: {len(normalized_data['TEMP'])}")
            logger.info(f"Normalized data lengths for EDA: {len(normalized_data['EDA'])}")
            logger.info(f"Label length: {len(data['label'])}")
            logger.info(f"Label shape: {data['label'].shape}")

            # Create and process DataFrame
            df = pd.DataFrame(normalized_data)
            logger.info(f"DataFrame columns: {df.columns}")
            logger.info(f"Length of BVP data: {len(df['BVP'])}")
            logger.info(f"DataFrame shape: {df.shape}")
            df = self.handle_missing_values(df, subject_id)

            logger.info(f"DataFrame shape after handling missing values: {df.shape}")

            # Process labels
            labels_final = self.process_labels(data['label'], len(df))

            return df, labels_final
            
        except Exception as e:
            logger.error(f"Error processing subject {subject_id}: {str(e)}")
            raise

    def handle_missing_values(self, df: pd.DataFrame, subject_id: int) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame using linear interpolation.

        Args:
            df (pd.DataFrame): DataFrame containing sensor data.
            subject_id (int): Subject identifier.

        Returns:
            pd.DataFrame: DataFrame with missing values handled.
        """
        if df.isnull().any().any():
            logger.warning(f"Missing values detected for subject {subject_id}")
            return df.interpolate(method='linear')
        return df
    
    def normalize_signal_lengths(self, data: Dict) -> Dict:
        """
        Normalize all signals to target length based on common sampling rate.

        Args:
            data (Dict): Dictionary containing wrist sensor data.

        Returns:
            Dict: Dictionary with normalized signal lengths.
        """
        target_length = int(len(data['label']) * (self.config.common_sampling_rate / 700))
        normalized_data = {}
        
        for signal, rate in self.config.sampling_rates['wrist'].items():
            if signal == 'ACC':
                acc_data = data['signal']['wrist']['ACC']
                logger.info(f"ACC shape: {acc_data.shape}")
                for i, axis in enumerate(['X', 'Y', 'Z']):
                    temp_series = pd.Series(acc_data[:, i].flatten())  # Flatten array
                    window = int(rate / self.config.common_sampling_rate)
                    resampled = self.apply_rolling_mean(temp_series, window)
                    normalized_data[f'ACC_{axis}'] = resampled.iloc[::window][:target_length].values
            else:
                signal_data = data['signal']['wrist'][signal]
                logger.info(f"{signal} shape: {signal_data.shape}")
                signal_data = signal_data.flatten() if len(signal_data.shape) > 1 else signal_data
                    
                temp_series = pd.Series(signal_data)
                window = int(rate / self.config.common_sampling_rate)
                resampled = self.apply_rolling_mean(temp_series, window)
                normalized_data[signal] = resampled.iloc[::window][:target_length].values
            
            # Verify length
            if signal == 'ACC':
                for axis in ['X', 'Y', 'Z']:
                    if len(normalized_data[f'ACC_{axis}']) != target_length:
                        raise ValueError(f"Length mismatch for ACC_{axis}: Expected {target_length}, got {len(normalized_data[f'ACC_{axis}'])}")
            else:
                if len(normalized_data[signal]) != target_length:
                    raise ValueError(f"Length mismatch for {signal}: Expected {target_length}, got {len(normalized_data[signal])}")
        
        return normalized_data

    @staticmethod
    def apply_rolling_mean(series: pd.Series, window: int) -> pd.Series:
        """
        Apply rolling mean to a time series.

        Args:
            series (pd.Series): Time series data.
            window (int): Window size for rolling mean.

        Returns:
            pd.Series: Time series with rolling mean applied.
        """
        return series.rolling(window=window, min_periods=1, center=True).mean()

    def process_labels(self, labels: np.ndarray, target_length: int) -> np.ndarray:
        """
        Process labels from 700Hz to 4Hz (common sampling rate) sampling rate.
        
        Args:
            labels (np.ndarray): Raw labels at 700Hz.
            target_length (int): Required length at common sampling rate.

        Returns:
            np.ndarray: Processed binary labels.
        """
        downsample_factor = int(700 // self.config.common_sampling_rate)
        logger.info(f"Downsample factor: {downsample_factor}")

        labels_df = pd.DataFrame(labels, columns=['label'])
        
        labels_resampled = labels_df.rolling(
            window=int(downsample_factor),
            min_periods=int(downsample_factor * 0.8),  # 80% minimum samples
            center=True
        ).apply(lambda x: np.bincount(x.astype(int)).argmax())

        logger.info(f"Labels resampled shape: {labels_resampled.shape}")
        
        labels_resampled = labels_resampled.dropna().reset_index(drop=True)  # Drop NaNs and reset index
        labels_downsampled = labels_resampled.iloc[::downsample_factor].reset_index(drop=True)

        logger.info(f"Labels downsampled shape: {labels_downsampled.shape}")
        
        labels_binary = (labels_downsampled == 2).astype(int)
        
        if len(labels_binary) < target_length:
            raise ValueError(f"Insufficient labels after processing: {len(labels_binary)} < {target_length}")
        
        return labels_binary[:target_length].to_numpy()

    def process_all_subject_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Process data for all valid subjects.
        
        Returns:
            Tuple[pd.DataFrame, np.ndarray]: Combined processed data and labels.
        """
        all_data = []
        all_labels = []
        
        for subject_id in self.config.valid_subjects:
            try:
                data, labels = self.extract_wrist_data(subject_id)
                all_data.append(data)
                all_labels.append(labels)
                logger.info(f"Successfully processed subject {subject_id}")
            except Exception as e:
                logger.error(f"Error processing subject {subject_id}: {str(e)}")
                continue
        
        if not all_data:
            raise ValueError("No data was successfully processed")

        combined_data = pd.concat(all_data, axis=0, ignore_index=True)
        combined_labels = np.concatenate(all_labels)
        
        return combined_data, combined_labels

    def save_processed_data(self) -> None:
        """
        Save processed data and labels to CSV file.
        """
        try:
            data, labels = self.process_all_subject_data()
            output_dir = Path(self.config.data_target_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            data["stress_label"] = labels
            output_file = output_dir / "processed_data.csv"
            data.to_csv(output_file, index=False)

            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise

    def log_data_statistics(self, data: pd.DataFrame, labels: np.ndarray) -> None:
        """
        Log statistics about the processed data.

        Args:
            data (pd.DataFrame): Processed sensor data.
            labels (np.ndarray): Processed binary labels.
        """
        logger.info(f"Total samples: {len(data)}")
        logger.info(f"Features: {list(data.columns[:-1])}")
        logger.info(f"Class distribution: {np.bincount(labels)}")
        logger.info(f"Class balance ratio: {np.bincount(labels)[1] / len(labels):.2%}")
