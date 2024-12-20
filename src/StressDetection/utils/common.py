# Standard library imports
import os  # Provides functions to interact with the operating system
from pathlib import Path  # Provides classes to handle filesystem paths

# Third-party library imports
import yaml  # Used for parsing and writing YAML
import json  # Used for parsing and writing JSON
from box import ConfigBox  # Provides a dictionary-like object with dot notation access
from box.exceptions import BoxValueError  # Exception handling for ConfigBox
from ensure import ensure_annotations  # Provides runtime type checking
from typing import Any  # Used for type hinting
import joblib  # Used for saving and loading binary files

# Local application/library specific imports
from StressDetection import logger  # Custom logger for the StressDetection application


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns its contents as a ConfigBox object.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Raises:
        ValueError: If the YAML file is empty.
        Exception: If there is an error reading the file.

    Returns:
        ConfigBox: The contents of the YAML file as a ConfigBox object.
    """
    try:
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file) or {}
            if not content:
                raise ValueError("YAML file is empty")
            logger.info(f"YAML file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise ValueError("YAML file is empty") from e
    except Exception as e:
        logger.error(f"Error reading YAML file: {e}")
        raise e
    

@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True) -> None:
    """Creates a list of directories if they do not already exist.

    Args:
        path_to_directories (list): List of paths of directories to create.
        verbose (bool, optional): If True, logs the creation of each directory. Defaults to True.
    """
    for path in path_to_directories:
        try:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"Created directory at: {path}")
        except Exception as e:
            logger.error(f"Error creating directory at {path}: {e}")
            raise

@ensure_annotations
def save_pkl_file(data: Any, path: Path):
    """save binary pkl file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")

