"""Central configuration for the Playlist Curator project.

Every script imports its paths, column names and settings from this file,
so changing something here changes it everywhere.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Folders and file paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_FILE_PATTERN = "*_music_data.csv"  # matches blues_music_data.csv, etc.

TRAIN_CSV = PROCESSED_DIR / "train.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"

MODEL_PATHS = {
    "logistic_regression": MODELS_DIR / "logistic_regression.joblib",
    "svm_rbf": MODELS_DIR / "svm_rbf.joblib",
    "neural_net": MODELS_DIR / "neural_net.pt",
}

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "svm_rbf": "RBF SVM",
    "neural_net": "Neural Network (PyTorch)",
}

# --------------------------------------------------------------------------
# Columns
# --------------------------------------------------------------------------
# The raw CSVs use these headers; we rename them to simple lowercase names.
RAW_COLUMN_RENAMES = {
    "Artist Name": "artist",
    "Track Name": "track",
    "Genres": "genres",
    "Playlist": "playlist",
}

ID_COL = "id"
TARGET_COL = "playlist"
GENRE_COL = "genres"

# The 11 audio features, in the order they are stored in the processed CSVs.
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

# How each audio feature is preprocessed:
CONTINUOUS_FEATURES = [  # standardised (zero mean, unit variance)
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]
BINARY_FEATURES = ["mode"]  # already 0 (minor) / 1 (major)
KEY_FEATURES = ["key"]  # pitch class 0-11 -> one-hot encoded
KEY_VALUES = list(range(12))
KEY_NAMES = ["C", "C#/Db", "D", "D#/Eb", "E", "F",
             "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"]

# Columns that the preprocessor expects as input.
MODEL_INPUT_COLUMNS = AUDIO_FEATURES + [GENRE_COL]

# Columns kept in data/processed/train.csv and test.csv.
CLEAN_COLUMNS = [ID_COL, "artist", "track", GENRE_COL, TARGET_COL] + AUDIO_FEATURES

# --------------------------------------------------------------------------
# Experiment settings
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20  # 80 % train / 20 % test
VAL_SIZE = 0.10   # share of the training set used for NN early stopping
GENRE_TOP_N = 50  # keep the 50 most common genre tags as features

# The dataset contains 347 overlapping playlists, far too many (and too
# similar) to learn from audio features alone.  We therefore model one
# listener's "taste" as a handful of clearly different playlists.
# Edit this list to build a different taste profile.
# (Run `python -m src.preprocess --list-playlists` to see every name.)
SELECTED_PLAYLISTS = [
    "Deathcore",
    "Metal Ballads",
    "Hard Rock",
    "Yacht Rock",
    "Acoustic Hits",
    "Retro Pop",
    "Jazz Rap",
    "Lo-fi Indie",
    "Shoegaze Classics",
    "Acoustic Blues",
]

# --------------------------------------------------------------------------
# Model hyper-parameters
# --------------------------------------------------------------------------
LOGREG_PARAMS = {"C": 1.0, "max_iter": 2000, "class_weight": "balanced"}

SVM_PARAMS = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "class_weight": "balanced",
    "probability": True,  # needed so the app can show confidence scores
    "random_state": RANDOM_STATE,
}

# The reference paper found one hidden layer beat two hidden layers (82 % vs 77 %
# test accuracy), so the default is a single hidden layer.  For a two-layer
# comparison like the paper's Table 3, try "hidden_dims": (64, 32).
NN_PARAMS = {
    "hidden_dims": (64,),
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "max_epochs": 200,
    "patience": 20,  # stop if validation loss does not improve for 20 epochs
    "seed": RANDOM_STATE,
}

# --------------------------------------------------------------------------
# Streamlit app: slider ranges (min, max, default)
# --------------------------------------------------------------------------
FEATURE_UI = {
    "danceability": (0.0, 1.0, 0.55),
    "energy": (0.0, 1.0, 0.70),
    "loudness": (-40.0, 5.0, -7.0),
    "speechiness": (0.0, 1.0, 0.05),
    "acousticness": (0.0, 1.0, 0.15),
    "instrumentalness": (0.0, 1.0, 0.00),
    "liveness": (0.0, 1.0, 0.15),
    "valence": (0.0, 1.0, 0.50),
    "tempo": (0.0, 260.0, 120.0),
}
