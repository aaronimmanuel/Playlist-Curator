"""Feature engineering: scaling, key one-hot encoding and genre encoding.

Everything is bundled into ONE scikit-learn ColumnTransformer so that the exact
same steps are applied during training, evaluation and in the Streamlit app.
"""

import ast

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def parse_genres(value) -> list:
    """Turn a genre cell into a clean list of lowercase tags.

    Accepts "['rock', 'modern rock']" (as stored in the CSVs), a real list,
    or a plain comma-separated string.  Missing values give an empty list.
    """
    if isinstance(value, (list, tuple)):
        tags = list(value)
    elif isinstance(value, str) and value.strip():
        try:
            tags = ast.literal_eval(value)
            if not isinstance(tags, (list, tuple)):
                tags = [str(tags)]
        except (ValueError, SyntaxError):
            tags = value.split(",")
    else:
        tags = []
    return [str(tag).strip().lower() for tag in tags if str(tag).strip()]


def build_preprocessor() -> ColumnTransformer:
    """Create the (unfitted) feature preprocessor.

    Output columns, in order:
      1. 9 continuous audio features, standardised
      2. mode (0/1), passed through unchanged
      3. key, one-hot encoded into 12 columns
      4. genre, multi-hot encoded using the top-N most frequent genre tags
    """
    return ColumnTransformer(
        transformers=[
            ("audio", StandardScaler(), config.CONTINUOUS_FEATURES),
            ("mode", "passthrough", config.BINARY_FEATURES),
            (
                "key",
                OneHotEncoder(categories=[config.KEY_VALUES], handle_unknown="ignore"),
                config.KEY_FEATURES,
            ),
            (
                "genre",
                CountVectorizer(
                    analyzer=parse_genres,
                    max_features=config.GENRE_TOP_N,
                    binary=True,
                ),
                config.GENRE_COL,  # a single column name -> 1-D input for CountVectorizer
            ),
        ],
        sparse_threshold=0.0,  # always return a normal dense numpy array
    )


def get_genre_vocabulary(preprocessor: ColumnTransformer) -> list:
    """Return the genre tags the fitted preprocessor knows about."""
    genre_encoder = preprocessor.named_transformers_["genre"]
    return list(genre_encoder.get_feature_names_out())
