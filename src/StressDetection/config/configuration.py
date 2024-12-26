from pathlib import Path
from StressDetection.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from StressDetection.utils.common import read_yaml, create_directories
from StressDetection.entity.entity import WesadDataIngestionConfig

class ConfigurationManager:
    """
    ConfigurationManager handles the configuration settings for the StressDetection project.
    """

    def __init__(self, config_file_path: Path = Path(CONFIG_FILE_PATH), params_file_path: Path = Path(PARAMS_FILE_PATH)):
        """
        Initialize ConfigurationManager with paths to the configuration and parameters files.

        :param config_file_path: Path to the configuration YAML file.
        :param params_file_path: Path to the parameters YAML file.
        """
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
    

    def get_wesad_data_ingestion_config(self) -> WesadDataIngestionConfig:
        """
        Get the configuration for WESAD data ingestion.

        :return: WesadDataIngestionConfig object with the necessary configuration settings.
        """
        config = self.config.wesad_data_ingestion
        params = self.params
        return WesadDataIngestionConfig(
            data_source_path=config.data_source_path,
            data_target_path=config.data_target_path,
            valid_subjects=params.valid_subjects,
            required_signals=params.required_signals,
            sampling_rates=params.sampling_rates,
            common_sampling_rate=params.common_sampling_rate,
            window_size=params.window_size
        )
    
    