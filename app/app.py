"""Streamlit app: describe a song and get the playlist it fits best.

Run from the project root:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

# Make `import src...` work when Streamlit runs this file from the app/ folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src import config
from src.data_loader import load_processed_split
from src.features import parse_genres
from src.predict import PlaylistCurator

st.set_page_config(page_title="Playlist Curator", page_icon="🎧", layout="wide")


# --------------------------------------------------------------------------
# Cached loaders (run once, not on every click)
# --------------------------------------------------------------------------
@st.cache_resource
def get_curator() -> PlaylistCurator:
    return PlaylistCurator()


@st.cache_data
def get_test_songs() -> pd.DataFrame:
    return load_processed_split("test")


# --------------------------------------------------------------------------
# Widget defaults (stored in st.session_state so a button can overwrite them)
# --------------------------------------------------------------------------
def init_defaults() -> None:
    for feature, (_, _, default) in config.FEATURE_UI.items():
        st.session_state.setdefault(feature, float(default))
    st.session_state.setdefault("key", 0)
    st.session_state.setdefault("mode", 1)
    st.session_state.setdefault("genres", [])
    st.session_state.setdefault("example_note", "")


def load_random_test_song() -> None:
    """Button callback: copy a random test-set song into the input widgets."""
    song = get_test_songs().sample(n=1).iloc[0]
    known_genres = set(get_curator().genre_vocabulary)

    for feature, (low, high, _) in config.FEATURE_UI.items():
        st.session_state[feature] = float(min(max(round(song[feature], 2), low), high))
    st.session_state["key"] = int(song["key"])
    st.session_state["mode"] = int(song["mode"])
    st.session_state["genres"] = [
        tag for tag in parse_genres(song["genres"]) if tag in known_genres
    ]
    st.session_state["example_note"] = (
        f"Loaded: **{song['artist']} - {song['track']}** "
        f"(its real playlist: **{song['playlist']}**)"
    )


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.title("🎧 Playlist Curator")
st.write("Enter a song's audio features and genre tags. The trained model predicts "
         "which of your playlists it belongs in.")

try:
    curator = get_curator()
except FileNotFoundError as error:
    st.error(f"{error}\n\nRun the preprocessing and training steps first (see README.md).")
    st.stop()

init_defaults()

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Model",
        options=list(config.MODEL_PATHS),
        format_func=lambda name: config.MODEL_DISPLAY_NAMES[name],
    )
    st.caption("Your playlists:")
    for playlist in curator.playlists:
        st.write(f"- {playlist}")

st.button("🎲 Fill in a random song from the test set", on_click=load_random_test_song)
if st.session_state["example_note"]:
    st.info(st.session_state["example_note"])

left, right = st.columns(2)
features = {}

with left:
    st.subheader("Audio features")
    for feature, (low, high, _) in config.FEATURE_UI.items():
        features[feature] = st.slider(feature.capitalize(), min_value=low, max_value=high,
                                      step=0.01, key=feature)

with right:
    st.subheader("Key, mode and genre")
    features["key"] = st.selectbox(
        "Key", options=config.KEY_VALUES, key="key",
        format_func=lambda k: f"{config.KEY_NAMES[k]} ({k})",
    )
    features["mode"] = st.radio(
        "Mode", options=[1, 0], key="mode", horizontal=True,
        format_func=lambda m: "Major" if m == 1 else "Minor",
    )
    features["genres"] = st.multiselect(
        "Genre tags", options=curator.genre_vocabulary, key="genres",
        help="The most common genre tags in the training data. Leave empty if unknown.",
    )

    predict_clicked = st.button("Predict playlist", type="primary")

# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------
st.divider()
if predict_clicked:
    ranking = curator.predict(features, model_name, top_k=len(curator.playlists))
    best_name, best_prob = ranking[0]

    st.success(f"Best match: **{best_name}**  ({best_prob:.0%} confidence)")

    result_left, result_right = st.columns(2)
    with result_left:
        st.subheader("Top 3 playlists")
        top3 = pd.DataFrame(ranking[:3], columns=["Playlist", "Probability"])
        top3["Probability"] = top3["Probability"].map("{:.1%}".format)
        st.table(top3.set_index("Playlist"))
    with result_right:
        st.subheader("All playlists")
        st.bar_chart(pd.DataFrame(ranking, columns=["Playlist", "Probability"]).set_index("Playlist"))

    with st.expander("Compare all three models"):
        rows = []
        for name in config.MODEL_PATHS:
            top_name, top_prob = curator.predict(features, name, top_k=1)[0]
            rows.append({"Model": config.MODEL_DISPLAY_NAMES[name],
                         "Predicted playlist": top_name,
                         "Confidence": f"{top_prob:.1%}"})
        st.table(pd.DataFrame(rows).set_index("Model"))
else:
    st.caption("Adjust the inputs, then press **Predict playlist**.")
