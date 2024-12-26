from StressDetection.config.configuration import ConfigurationManager
from StressDetection.components.data_ingestion import WESADLoader
from StressDetection import logger

STAGE_NAME = "DATA_INGESTION"

class WesadDataIngestionPipeline:
    def __init__(self) -> None:
        """Initialize the WesadDataIngestionPipeline."""
        pass

    def main(self) -> None:
        """Main method to run the data ingestion pipeline."""
        config = ConfigurationManager()
        wesad_data_ingestion_config = config.get_wesad_data_ingestion_config()
        wesad_data_ingestion = WESADLoader(config=wesad_data_ingestion_config)
        wesad_data_ingestion.save_processed_data()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = WesadDataIngestionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e