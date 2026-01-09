import re
from collections import Counter
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer


# -----------------------------
# Tag cleaning helpers
# -----------------------------
_BAD_TAGS = {
    "seen", "owned", "want", "favorite", "favourite", "boring",
    "dvd", "vhs", "cd", "soundtrack", "netflix", "amazon", "prime",
    "watchlist", "to watch", "rewatch", "re-watch", "old"
}

def _clean_tag(tag: str) -> str | None:
    """Normalize + filter noisy MovieLens tags."""
    if not isinstance(tag, str):
        return None

    t = tag.strip().lower()
    t = re.sub(r"[^a-z0-9\s\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    if not t or len(t) < 3 or len(t) > 30:
        return None
    if t.isdigit():
        return None
    if t in _BAD_TAGS:
        return None

    return t


def prepare_data(
    ratings_path: str = "ml-latest-small/ratings.csv",
    movies_path: str = "ml-latest-small/movies.csv",
    tags_path: str = "ml-latest-small/tags.csv",
    *,
    # Tag controls
    top_n_tags_per_movie: int = 8,
    min_global_tag_count: int = 5,
    max_tag_vocab: int = 3000,
) -> Tuple[pd.DataFrame, pd.DataFrame, int, int, torch.Tensor, csr_matrix, Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Prepare MovieLens data for LightGCN + Streamlit app.

    Returns
    -------
    ratings:
        Original ratings + added columns:
        - user_id (internal contiguous index)
        - movie_id (internal contiguous index)

    movies_out:
        Movie metadata aligned to internal movie_id order (critical!):
        - movie_id (internal index)
        - movieId (original MovieLens ID)
        - title, genres
        - tag_raw (raw aggregated tags)
        - tag (cleaned + filtered tags)

    edge_index:
        Bidirectional edges for a user–item bipartite graph in a single node ID space:
        - user nodes are [0 .. num_users-1]
        - item nodes are [num_users .. num_users+num_items-1]
        Shape: [2, num_edges * 2]  (u->i and i->u)

    tag_features:
        Bag-of-words CSR matrix aligned to internal movie_id (row i == movie_id i)

    movie_id_to_index:
        Mapping from original movieId -> internal movie_id (for compatibility)

    user2id, movie2id:
        Original ID -> internal ID mappings
    """
    # -----------------------------
    # 1) Load raw CSVs
    # -----------------------------
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)   # movieId, title, genres
    tags = pd.read_csv(tags_path)

    # -----------------------------
    # 2) Create contiguous ID mappings (from ratings, so graph is consistent)
    # -----------------------------
    user2id = {u: i for i, u in enumerate(ratings["userId"].unique())}
    movie2id = {m: i for i, m in enumerate(ratings["movieId"].unique())}

    ratings["user_id"] = ratings["userId"].map(user2id)
    ratings["movie_id"] = ratings["movieId"].map(movie2id)

    num_users = len(user2id)
    num_items = len(movie2id)

    # -----------------------------
    # 3) Movies table aligned to internal movie_id order (IMPORTANT)
    # -----------------------------
    id2movie = {idx: mid for mid, idx in movie2id.items()}
    movies_index = pd.DataFrame({
        "movie_id": np.arange(num_items, dtype=int),
        "movieId": [id2movie[i] for i in range(num_items)]
    })

    # Merge titles/genres
    movies_out = movies_index.merge(
        movies[["movieId", "title", "genres"]],
        on="movieId",
        how="left"
    )

    # -----------------------------
    # 4) Tags: raw aggregation + cleaned reduction
    # -----------------------------
    # Raw tags per movie (for “before vs after” in the app)
    if not tags.empty and "tag" in tags.columns:
        raw_tags = (
            tags.groupby("movieId")["tag"]
            .apply(lambda x: " ".join(map(str, x.tolist())))
            .reset_index(name="tag_raw")
        )
    else:
        raw_tags = pd.DataFrame({"movieId": movies_out["movieId"], "tag_raw": ""})

    # Clean tags
    if not tags.empty and "tag" in tags.columns:
        tags = tags.copy()
        tags["tag_clean"] = tags["tag"].apply(_clean_tag)
        tags = tags.dropna(subset=["tag_clean"])

        # Drop globally rare tags
        global_counts = tags["tag_clean"].value_counts()
        keep = set(global_counts[global_counts >= min_global_tag_count].index)
        tags = tags[tags["tag_clean"].isin(keep)]

        # Count per movie and keep top N tags per movie
        movie_tag_counts = (
            tags.groupby(["movieId", "tag_clean"])
            .size()
            .reset_index(name="cnt")
            .sort_values(["movieId", "cnt"], ascending=[True, False])
        )
        top_movie_tags = movie_tag_counts.groupby("movieId").head(top_n_tags_per_movie)

        clean_tags = (
            top_movie_tags.groupby("movieId")["tag_clean"]
            .apply(lambda x: " ".join(x.tolist()))
            .reset_index(name="tag")
        )
    else:
        clean_tags = pd.DataFrame({"movieId": movies_out["movieId"], "tag": ""})

    # Merge tags into movies_out
    movies_out = movies_out.merge(raw_tags, on="movieId", how="left")
    movies_out = movies_out.merge(clean_tags, on="movieId", how="left")
    movies_out["tag_raw"] = movies_out["tag_raw"].fillna("")
    movies_out["tag"] = movies_out["tag"].fillna("")

    # For UI friendliness
    movies_out["tag_raw"] = movies_out["tag_raw"].replace("", "No tags available")
    movies_out["tag"] = movies_out["tag"].replace("", "No tags available")

    # -----------------------------
    # 5) Build bidirectional edge_index (user–movie bipartite graph)
    # -----------------------------
    user_nodes = ratings["user_id"].to_numpy(dtype=np.int64)
    item_nodes = (ratings["movie_id"].to_numpy(dtype=np.int64) + num_users)

    u2i = np.vstack([user_nodes, item_nodes])
    i2u = np.vstack([item_nodes, user_nodes])

    edge_index = torch.tensor(
        np.hstack([u2i, i2u]),
        dtype=torch.long
    )

    # -----------------------------
    # 6) Build tag features aligned to internal movie_id
    # -----------------------------
    vectorizer = CountVectorizer(
        max_features=max_tag_vocab,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b",
    )
    tag_features = vectorizer.fit_transform(movies_out["tag"].astype(str).tolist())
    tag_features = csr_matrix(tag_features)

    # Compatibility mapping (original movieId -> internal movie_id)
    movie_id_to_index = movie2id

    print("✅ Data prepared successfully!")
    print(f"   👥 Users: {num_users}")
    print(f"   🎬 Movies: {num_items}")
    print(f"   ⭐ Ratings: {len(ratings)}")
    print(f"   🔗 Graph edges (bidirectional): {edge_index.shape[1]}")
    print(f"   🏷️ Tag features: {tag_features.shape} (movies x vocab)")

    return (
        ratings,
        movies_out[["movie_id", "movieId", "title", "genres", "tag_raw", "tag"]].copy(),
        num_users,
        num_items,
        edge_index,
        tag_features,
        movie_id_to_index,
        user2id,
        movie2id,
    )
