# ================================================
# prediction.py  |  LightGCN Recommender Utilities
# ================================================
import re
import time
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix

from data_preparation import prepare_data
from model_training import LightGCN


# -----------------------------
# Load Data & Model Once
# -----------------------------
def load_model_and_data():
    """
    Loads:
      - MovieLens data prepared for LightGCN (graph + cleaned tags)
      - LightGCN checkpoint (if available)
      - Precomputed user/item embeddings
    """
    start_time = time.time()

    (
        ratings,
        movies,
        num_users,
        num_items,
        edge_index,
        tag_features,
        movie_id_to_index,
        user2id,
        movie2id,
    ) = prepare_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = "lightgcn_checkpoint.pth"
    history = None
    k_eval = 10

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        embed_dim = int(checkpoint.get("embed_dim", 128))
        num_layers = int(checkpoint.get("num_layers", 3))
        history = checkpoint.get("history")
        k_eval = int(checkpoint.get("k_eval", 10))

        model = LightGCN(num_users, num_items, embed_dim=embed_dim, num_layers=num_layers)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device).eval()

        print(f"✅ Loaded checkpoint: {checkpoint_path} (dim={embed_dim}, layers={num_layers})")
    except FileNotFoundError:
        model = LightGCN(num_users, num_items, embed_dim=128, num_layers=3).to(device).eval()
        print("⚠️ No checkpoint found. Train first to get meaningful recommendations.")

    with torch.no_grad():
        user_embs, item_embs = model(edge_index.to(device), num_users, num_items)

    user_embs = user_embs.cpu()
    item_embs = item_embs.cpu()

    print(f"✅ Ready in {time.time() - start_time:.2f}s (users={num_users}, items={num_items})")

    return (
        ratings,
        movies,
        num_users,
        num_items,
        edge_index,
        tag_features,
        movie_id_to_index,
        user2id,
        movie2id,
        user_embs,
        item_embs,
        history,
        k_eval,
    )


# -----------------------------
# Display User Ratings
# -----------------------------
def show_user_ratings(user_id: int, ratings: pd.DataFrame, movies: pd.DataFrame):
    if user_id not in ratings["userId"].values:
        return (
            pd.DataFrame(columns=["title", "rating", "tag"]),
            pd.DataFrame(columns=["title", "rating", "tag"]),
        )

    user_r = ratings[ratings["userId"] == user_id].merge(
        movies, left_on="movie_id", right_on="movie_id", how="left"
    )

    high_rated = (
        user_r[user_r["rating"] >= 4][["title", "rating", "tag"]]
        .sort_values(by="rating", ascending=False)
    )
    low_rated = (
        user_r[user_r["rating"] < 4][["title", "rating", "tag"]]
        .sort_values(by="rating", ascending=True)
    )
    return high_rated, low_rated



# -----------------------------
# Recommend Movies
# -----------------------------
def recommend_movies_for_user(
    user_id: int,
    user_embs: torch.Tensor,
    item_embs: torch.Tensor,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    tag_features: csr_matrix,
    user2id: Dict[int, int],
    movie2id: Dict[int, int],
    movie_id_to_index: Dict[int, int],
    top_k: int = 10,
    alpha: float = 0.30,
):
    """
    Hybrid ranking:
      final_score = (1 - alpha) * LightGCN_score + alpha * tag_similarity_score

    Returns a dataframe with score breakdown so your Streamlit app can explain
    *why* a movie was recommended (useful for recruiters).
    """
    if user_id not in user2id:
        return pd.DataFrame(columns=["movie_id", "title", "genres", "tag", "score_gnn", "score_tag", "score_final"])

    start = time.time()
    user_idx = user2id[user_id]

    # --- LightGCN score (graph-based CF) ---
    score_gnn = (user_embs[user_idx] @ item_embs.T).float()

    # --- Content score (tag similarity) ---
    score_tag = torch.zeros(item_embs.shape[0], dtype=torch.float32)

    user_r = ratings[ratings["user_id"] == user_idx]
    rated_item_ids = user_r["movie_id"].to_list()

    if rated_item_ids:
        user_profile = tag_features[rated_item_ids].mean(axis=0)  # 1 x vocab
        user_profile = csr_matrix(user_profile)
        sim = user_profile.dot(tag_features.T)                    # 1 x items
        score_tag_np = sim.toarray().ravel().astype(np.float32)
        score_tag = torch.from_numpy(score_tag_np)


    score_final = (1.0 - alpha) * score_gnn + alpha * score_tag

    # mask already rated
    if rated_item_ids:
        score_final[rated_item_ids] = -float("inf")

    top_items = torch.topk(score_final, top_k).indices.tolist()

    cols = ["movie_id", "title", "genres", "tag"]
    available = [c for c in cols if c in movies.columns]
    top_df = movies.iloc[top_items][available].reset_index(drop=True)

    # attach score breakdown
    top_df["score_gnn"] = score_gnn[top_items].detach().cpu().numpy()
    top_df["score_tag"] = score_tag[top_items].detach().cpu().numpy()
    top_df["score_final"] = score_final[top_items].detach().cpu().numpy()

    print(f"✅ Recommendations generated in {time.time() - start:.2f}s")
    return top_df


# -----------------------------
# Helper for Streamlit
# -----------------------------
def generate_recommendations(user_id: int, data_bundle, top_k: int = 10, alpha: float = 0.30):
    (
        ratings,
        movies,
        _num_users,
        _num_items,
        _edge_index,
        tag_features,
        movie_id_to_index,
        user2id,
        movie2id,
        user_embs,
        item_embs,
        _history,
        _k_eval,
    ) = data_bundle

    high_rated, low_rated = show_user_ratings(user_id, ratings, movies)
    recs = recommend_movies_for_user(
        user_id,
        user_embs,
        item_embs,
        ratings,
        movies,
        tag_features,
        user2id,
        movie2id,
        movie_id_to_index,
        top_k=top_k,
        alpha=alpha,
    )
    return high_rated, low_rated, recs


# -----------------------------
# Tag suggestion (uses embedding neighbors)
# -----------------------------
def _tokenize_tags(tag_str: str) -> List[str]:
    if not isinstance(tag_str, str):
        return []
    if tag_str.lower().strip() == "no tags available":
        return []
    parts = re.split(r"[\s,]+", tag_str.lower())
    return [p.strip() for p in parts if p.strip()]


def suggest_tags_from_neighbors(
    movie_internal_id: int,
    movies: pd.DataFrame,
    item_embs: torch.Tensor,
    *,
    k_neighbors: int = 25,
    top_tags: int = 8,
) -> List[str]:
    """
    Suggest tags for a movie by looking at nearest movies in the learned embedding space.
    Practical denoising: "similar movies tend to share meaningful tags".
    """
    if movie_internal_id < 0 or movie_internal_id >= item_embs.shape[0]:
        return []

    emb = item_embs[movie_internal_id]
    sims = (item_embs @ emb).detach().cpu().numpy()
    nn = np.argsort(-sims)[1:k_neighbors + 1]  # skip itself

    bag: List[str] = []
    for j in nn:
        bag.extend(_tokenize_tags(str(movies.iloc[j].get("tag", ""))))

    counts = Counter(bag)
    return [t for t, _ in counts.most_common(top_tags)]
