"""Step 1: turn the raw Spotify CSVs into clean train/test tables.

Run from the project root:
    python -m src.preprocess                    # build the dataset
    python -m src.preprocess --list-playlists   # show all available playlists
"""

import argparse

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src import config
from src.data_loader import load_raw_data
from src.features import build_preprocessor, get_genre_vocabulary
from src.utils import make_console_safe


def resolve_duplicates(data: pd.DataFrame, key: str) -> pd.DataFrame:
    """Keep one row per track.

    * Same track listed twice in the SAME playlist (e.g. it appears in two CSV
      files) -> keep one copy.
    * Same track in DIFFERENT selected playlists -> its label is ambiguous for a
      single-label classifier, so the track is removed entirely.
    """
    data = data.drop_duplicates(subset=[key, config.TARGET_COL])
    playlists_per_track = data.groupby(key)[config.TARGET_COL].transform("nunique")
    return data[playlists_per_track == 1].copy()


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Select the taste playlists and remove bad / duplicated rows."""
    missing = [p for p in config.SELECTED_PLAYLISTS if p not in set(raw[config.TARGET_COL])]
    if missing:
        raise ValueError(
            f"These playlists from config.SELECTED_PLAYLISTS were not found: {missing}\n"
            "Run `python -m src.preprocess --list-playlists` to see valid names."
        )

    data = raw[raw[config.TARGET_COL].isin(config.SELECTED_PLAYLISTS)].copy()
    print(f"Rows in the {len(config.SELECTED_PLAYLISTS)} selected playlists: {len(data)}")

    # Missing values and broken audio analysis (tempo == 0 means "not analysed")
    data = data.dropna(subset=config.AUDIO_FEATURES + [config.TARGET_COL])
    data = data[data["tempo"] > 0].copy()
    data[config.GENRE_COL] = data[config.GENRE_COL].fillna("[]")

    # One row per track.  First by Spotify id, then by artist + title in case the
    # same song was uploaded under two different ids.
    data = resolve_duplicates(data, config.ID_COL)
    data["track_key"] = data["artist"].str.lower() + "||" + data["track"].str.lower()
    data = resolve_duplicates(data, "track_key")
    print(f"Rows after removing duplicates and conflicting tracks: {len(data)}")

    return data[config.CLEAN_COLUMNS].reset_index(drop=True)


def list_playlists() -> None:
    raw = load_raw_data()
    counts = raw.groupby(config.TARGET_COL)[config.ID_COL].nunique().sort_values(ascending=False)
    print(f"{len(counts)} playlists found (name: number of unique tracks)\n")
    for name, n_tracks in counts.items():
        print(f"{n_tracks:5d}  {name}")


def main() -> None:
    make_console_safe()
    parser = argparse.ArgumentParser(description="Preprocess the Spotify playlist data.")
    parser.add_argument("--list-playlists", action="store_true",
                        help="print every playlist name in the raw data and exit")
    args = parser.parse_args()

    if args.list_playlists:
        list_playlists()
        return

    for folder in (config.PROCESSED_DIR, config.MODELS_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data()
    print(f"Loaded {len(raw)} raw rows from {raw['source_file'].nunique()} files.")
    data = clean_data(raw)

    # Stratified split keeps the playlist proportions the same in train and test.
    # Each track appears only once, so no song can leak from train into test.
    train_df, test_df = train_test_split(
        data,
        test_size=config.TEST_SIZE,
        stratify=data[config.TARGET_COL],
        random_state=config.RANDOM_STATE,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Fit the label encoder and the feature preprocessor on the TRAINING data only.
    label_encoder = LabelEncoder().fit(train_df[config.TARGET_COL])
    preprocessor = build_preprocessor().fit(train_df[config.MODEL_INPUT_COLUMNS])

    train_df.to_csv(config.TRAIN_CSV, index=False)
    test_df.to_csv(config.TEST_CSV, index=False)
    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)
    joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)

    n_features = preprocessor.transform(train_df[config.MODEL_INPUT_COLUMNS].head(1)).shape[1]
    print("\nPlaylists (classes):")
    for name in label_encoder.classes_:
        n_train = (train_df[config.TARGET_COL] == name).sum()
        n_test = (test_df[config.TARGET_COL] == name).sum()
        print(f"  {name:<24} train={n_train:4d}  test={n_test:4d}")
    print(f"\nTrain rows: {len(train_df)} | Test rows: {len(test_df)}")
    print(f"Feature vector length: {n_features} "
          f"(9 scaled audio + 1 mode + 12 key one-hot + "
          f"{len(get_genre_vocabulary(preprocessor))} genre tags)")
    print(f"\nSaved: {config.TRAIN_CSV}\n       {config.TEST_CSV}\n"
          f"       {config.PREPROCESSOR_PATH}\n       {config.LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    main()
