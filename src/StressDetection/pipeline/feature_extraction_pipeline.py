from StressDetection.config.configuration import ConfigurationManager
from StressDetection.components.feature_extraction import FeatureExtractor
from StressDetection import logger
from pathlib import Path
import pandas as pd

STAGE_NAME = "FEATURE_EXTRACTION"

class FeatureExtractionPipeline:
    def __init__(self) -> None:
        """Initialize the FeatureExtractionPipeline."""
        pass

    def main(self) -> None:
        """Main method to run the feature extraction pipeline."""
        config = ConfigurationManager()
        feature_extraction_config = config.get_feature_extraction_config()

        # Load the processed segments and quality scores
        processed_data_path = config.get_preprocessing_config().processed_data_target_path
        segments_dir = Path(processed_data_path / "processed_signal_segments")
        quality_scores_path = Path(processed_data_path / "quality_scores.csv")

        # Load quality scores
        quality_scores = pd.read_csv(quality_scores_path)

        # Filter valid segments based on quality scores
        valid_segments = []
        for i, score in enumerate(quality_scores['quality_score']):
            if score >= feature_extraction_config.quality_threshold:
                segment_path = segments_dir / f"segment_{i}.csv"
                segment = pd.read_csv(segment_path)
                valid_segments.append(segment)

        # Log statistics about segment filtering
        total_segments = len(quality_scores)
        valid_segment_count = len(valid_segments)
        failed_segment_count = total_segments - valid_segment_count
        logger.info(f"Total segments: {total_segments}")
        logger.info(f"Segments passing quality threshold: {valid_segment_count}")
        logger.info(f"Segments failing quality threshold: {failed_segment_count}")

        # Initialize FeatureExtractor
        feature_extractor = FeatureExtractor(config=feature_extraction_config)

        # Extract features from valid segments
        all_features = []
        for segment in valid_segments:
            features = feature_extractor.extract_features(segment)
            # Include stress label in the features DataFrame
            features['stress_label'] = segment['stress_label'].iloc[0]
            all_features.append(features)

        # Combine all features into a single DataFrame
        all_features_df = pd.concat(all_features, ignore_index=True)

        # Save the extracted features
        output_dir = Path(feature_extraction_config.target_path / "features")
        output_dir.mkdir(parents=True, exist_ok=True)
        all_features_df.to_csv(output_dir / "extracted_features.csv", index=False)

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = FeatureExtractionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e