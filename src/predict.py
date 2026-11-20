"""Predict the best playlist for a new song.

Use it from Python:
    from src.predict import PlaylistCurator
    curator = PlaylistCurator()
    curator.predict({"danceability": 0.6, ..., "genres": ["rock", "pop"]}, "svm_rbf")

Or run a quick demo on random test-set songs:
    python -m src.predict
"""

import argparse

import pandas as pd

from src import config
from src.data_loader import load_preprocessing_artifacts, load_processed_split
from src.features import get_genre_vocabulary
from src.models import load_model
from src.utils import make_console_safe


class PlaylistCurator:
    """Loads the saved preprocessor, label encoder and all three trained models."""

    def __init__(self):
        self.preprocessor, self.label_encoder = load_preprocessing_artifacts()
        self.models = {name: load_model(name) for name in config.MODEL_PATHS}
        self.playlists = list(self.label_encoder.classes_)
        self.genre_vocabulary = get_genre_vocabulary(self.preprocessor)

    def _to_table(self, song: dict) -> pd.DataFrame:
        """Turn one song (a dict) into a one-row table with the expected columns."""
        row = {feature: song[feature] for feature in config.AUDIO_FEATURES}
        genres = song.get(config.GENRE_COL, [])
        row[config.GENRE_COL] = genres if isinstance(genres, str) else str(list(genres))
        return pd.DataFrame([row])[config.MODEL_INPUT_COLUMNS]

    def predict_proba(self, song: dict, model_name: str = "svm_rbf") -> dict:
        """Return {playlist name: probability} for every playlist."""
        features = self.preprocessor.transform(self._to_table(song)).astype("float32")
        probabilities = self.models[model_name].predict_proba(features)[0]
        return dict(zip(self.playlists, probabilities))

    def predict(self, song: dict, model_name: str = "svm_rbf", top_k: int = 3) -> list:
        """Return the top_k best playlists as [(name, probability), ...], best first."""
        scores = self.predict_proba(song, model_name)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]


def main() -> None:
    make_console_safe()
    parser = argparse.ArgumentParser(description="Demo: predict playlists for random test songs.")
    parser.add_argument("--n", type=int, default=5, help="how many random test songs to try")
    args = parser.parse_args()

    curator = PlaylistCurator()
    test_table = load_processed_split("test")
    sample = test_table.sample(n=min(args.n, len(test_table)), random_state=config.RANDOM_STATE)

    for _, row in sample.iterrows():
        print(f"\n{row['artist']} - {row['track']}")
        print(f"  true playlist: {row[config.TARGET_COL]}")
        song = row.to_dict()
        for name in config.MODEL_PATHS:
            best_name, best_prob = curator.predict(song, name, top_k=1)[0]
            print(f"  {config.MODEL_DISPLAY_NAMES[name]:<26} -> {best_name} ({best_prob:.0%})")


if __name__ == "__main__":
    main()
