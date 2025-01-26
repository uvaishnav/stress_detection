from StressDetection.config.configuration import ConfigurationManager
from StressDetection.components.signal_processing import SignalProcessor
from StressDetection import logger
from pathlib import Path
import pandas as pd
import os

STAGE_NAME = "SIGNAL_PROCESSING"

class SignalProcessingPipeline:
    def __init__(self) -> None:
        """Initialize the SignalProcessingPipeline."""
        pass

    def main(self) -> None:
        """Main method to run the signal processing pipeline."""
        config = ConfigurationManager()
        signal_processing_config = config.get_signal_processing_config()

        # Load the processed data
        processed_data_path = config.get_preprocessing_config().processed_data_path
        data = pd.read_csv(processed_data_path / "processed_data.csv")

        # Initialize SignalProcessor
        signal_processor = SignalProcessor(config=signal_processing_config)

        # Process signals
        valid_segments, quality_scores = signal_processor.process_signals(data)

        # Save the processed segments and quality scores
        target_path = Path(config.get_preprocessing_config().processed_data_target_path)
        output_dir = Path(target_path / "processed_signal_segments")
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, segment in enumerate(valid_segments):
            segment.to_csv(output_dir / f"segment_{i}.csv", index=False)

        pd.DataFrame(quality_scores, columns=["quality_score"]).to_csv(target_path / "quality_scores.csv", index=False)

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = SignalProcessingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e