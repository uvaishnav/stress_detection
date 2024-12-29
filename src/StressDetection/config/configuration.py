from pathlib import Path
from typing import Dict, Any
from StressDetection.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from StressDetection.utils.common import read_yaml, create_directories
from StressDetection.entity.entity import (
    WesadDataIngestionConfig,
    PreprocessingConfig,
    SignalProcessingConfig,
)

class ConfigurationManager:
    """
    ConfigurationManager handles the configuration settings for the StressDetection project.
    
    This class is responsible for:
    - Loading and validating configuration files
    - Providing access to specific configuration sections
    - Ensuring type safety and parameter validation
    """

    def __init__(
        self, 
        config_file_path: Path = Path(CONFIG_FILE_PATH), 
        params_file_path: Path = Path(PARAMS_FILE_PATH)
    ) -> None:
        """
        Initialize ConfigurationManager with paths to the configuration and parameters files.

        Args:
            config_file_path (Path): Path to the configuration YAML file
            params_file_path (Path): Path to the parameters YAML file

        Raises:
            FileNotFoundError: If either configuration file doesn't exist
            ValueError: If configuration files are empty or invalid
        """
        # Validate file paths
        if not config_file_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {config_file_path}")
        if not params_file_path.exists():
            raise FileNotFoundError(f"Parameters file not found at: {params_file_path}")

        # Load configuration files
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)

        # Validate configuration content
        if not self.config or not self.params:
            raise ValueError("Configuration files are empty or invalid")

    def get_wesad_data_ingestion_config(self) -> WesadDataIngestionConfig:
        """
        Get the configuration for WESAD data ingestion.

        Returns:
            WesadDataIngestionConfig: Configuration object for data ingestion

        Raises:
            KeyError: If required configuration keys are missing
            ValueError: If parameter values are invalid
        """
        try:
            config = self.config.wesad_data_ingestion
            params = self.params

            # Validate required parameters
            if not config.data_source_path or not config.data_target_path:
                raise ValueError("Data paths cannot be empty")

            # Convert paths to Path objects and ensure they exist
            source_path = Path(config.data_source_path)
            target_path = Path(config.data_target_path)

            # Validate numerical parameters
            if params.common_sampling_rate <= 0:
                raise ValueError("Sampling rate must be positive")
            if params.window_size <= 0:
                raise ValueError("Window size must be positive")

            return WesadDataIngestionConfig(
                data_source_path=source_path,
                data_target_path=target_path,
                valid_subjects=list(params.valid_subjects),  # Ensure it's a list
                required_signals=list(params.required_signals),
                sampling_rates=dict(params.sampling_rates),  # Ensure it's a dictionary
                common_sampling_rate=float(params.common_sampling_rate),
                window_size=int(params.window_size)
            )
        except AttributeError as e:
            raise KeyError(f"Missing required configuration parameter: {str(e)}")

    def get_signal_processing_config(self) -> SignalProcessingConfig:
        """
        Get the configuration for signal processing.

        Returns:
            SignalProcessingConfig: Configuration object for signal processing

        Raises:
            KeyError: If required configuration keys are missing
            ValueError: If parameter values are invalid
        """
        try:
            config = self.config.signal_processing
            params = self.params

            # Validate frequency parameters
            for param in [config.bvp_filter_low, config.bvp_filter_high, 
                         config.eda_filter_low, config.eda_filter_high]:
                if not isinstance(param, (int, float)) or param < 0:
                    raise ValueError("Filter frequencies must be non-negative numbers")

            return SignalProcessingConfig(
                bvp_filter_low=float(config.bvp_filter_low),
                bvp_filter_high=float(config.bvp_filter_high),
                eda_filter_low=float(config.eda_filter_low),
                eda_filter_high=float(config.eda_filter_high),
                acc_threshold=float(config.acc_threshold),
                motion_window=int(config.motion_window),
                window_size=int(params.window_size),
                common_sampling_rate=int(params.common_sampling_rate),
                overlap=float(config.overlap),
                quality_threshold=float(config.quality_threshold),
                physiological_ranges=dict(params.physiological_ranges)
            )
        except AttributeError as e:
            raise KeyError(f"Missing required signal processing parameter: {str(e)}")

    def get_preprocessing_config(self) -> PreprocessingConfig:
        """
        Get the configuration for data preprocessing.

        Returns:
            PreprocessingConfig: Configuration object for preprocessing

        Raises:
            KeyError: If required configuration keys are missing
            ValueError: If parameter values are invalid
        """
        try:
            config = self.config.preprocessing
            params = self.params.preprocessing

            # Validate path and parameters
            processed_path = Path(config.processed_data_path)
            if not isinstance(params.temp_baseline, (int, float)):
                raise ValueError("Temperature baseline must be a number")
            if params.artifact_threshold <= 0:
                raise ValueError("Artifact threshold must be positive")

            return PreprocessingConfig(
                processed_data_path=processed_path,
                temp_baseline=float(params.temp_baseline),
                artifact_threshold=float(params.artifact_threshold)
            )
        except AttributeError as e:
            raise KeyError(f"Missing required preprocessing parameter: {str(e)}")
        
        