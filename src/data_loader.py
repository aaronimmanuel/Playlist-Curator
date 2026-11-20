"""Functions for reading the raw Spotify CSVs and the processed train/test splits."""

import joblib
import pandas as pd

from src import config


def load_raw_data() -> pd.DataFrame:
    """Read every *_music_data.csv in data/raw/ and stack them into one table."""
    csv_files = sorted(config.RAW_DIR.glob(config.RAW_FILE_PATTERN))
    if not csv_files:
        raise FileNotFoundError(
            f"No files matching '{config.RAW_FILE_PATTERN}' found in {config.RAW_DIR}.\n"
            "Unzip the dataset and copy the 7 CSV files into data/raw/ first."
        )

    frames = []
    for path in csv_files:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data = data.rename(columns=config.RAW_COLUMN_RENAMES)
    return data


def load_processed_split(split: str) -> pd.DataFrame:
    """Load the cleaned (not yet scaled) train or test table."""
    path = {"train": config.TRAIN_CSV, "test": config.TEST_CSV}[split]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.preprocess` first."
        )
    # genres are stored as text such as "['rock', 'pop']", so read them as strings
    return pd.read_csv(path, dtype={config.GENRE_COL: str}, keep_default_na=False)


def load_preprocessing_artifacts():
    """Load the fitted feature preprocessor and label encoder."""
    for path in (config.PREPROCESSOR_PATH, config.LABEL_ENCODER_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m src.preprocess` first."
            )
    preprocessor = joblib.load(config.PREPROCESSOR_PATH)
    label_encoder = joblib.load(config.LABEL_ENCODER_PATH)
    return preprocessor, label_encoder


def load_xy(split: str):
    """Return (X, y) for 'train' or 'test': scaled/encoded features and integer labels."""
    preprocessor, label_encoder = load_preprocessing_artifacts()
    table = load_processed_split(split)
    X = preprocessor.transform(table[config.MODEL_INPUT_COLUMNS]).astype("float32")
    y = label_encoder.transform(table[config.TARGET_COL])
    return X, y
