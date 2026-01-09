# ================================================
# app.py  |  Streamlit App: LightGCN (GNN) Movie Recommender
# ================================================
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
from sklearn.decomposition import PCA

from model_training import LightGCN, train_lightgcn
from prediction import (
    generate_recommendations,
    load_model_and_data,
    suggest_tags_from_neighbors,
)


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="LightGCN Movie Recommender (GNN)", layout="wide")


# ===========================================================
# TMDB: optional poster enrichment
# ===========================================================
DEFAULT_POSTER = "./default.png"


def load_tmdb_key() -> str | None:
    key = os.getenv("TMDB_API_KEY")
    if key:
        return key.strip()
    try:
        with open("tmdb_key.txt", "r") as f:
            return f.read().strip()
    except Exception:
        return None


TMDB_API_KEY = load_tmdb_key()


def tmdb_search_movie(title: str, year: int | None = None):
    if not TMDB_API_KEY:
        return None
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title}
    if year:
        params["year"] = year
    try:
        r = requests.get(url, params=params, timeout=8)
        data = r.json()
        if data.get("results"):
            return data["results"][0]
        return None
    except Exception:
        return None


def tmdb_get_poster(tmdb_movie) -> str:
    if not tmdb_movie:
        return DEFAULT_POSTER
    poster_path = tmdb_movie.get("poster_path")
    if not poster_path:
        return DEFAULT_POSTER
    return f"https://image.tmdb.org/t/p/w500{poster_path}"


# ===========================================================
# Cached data
# ===========================================================
@st.cache_resource(show_spinner=True)
def get_data_bundle():
    return load_model_and_data()


@st.cache_data(show_spinner=False)
def compute_degrees(_edge_index, n_nodes: int) -> np.ndarray:
    deg = torch.bincount(_edge_index[0].cpu(), minlength=n_nodes).numpy()
    return deg



# ===========================================================
# Visualization helpers
# ===========================================================
def _short_title(t: str, max_len: int = 22) -> str:
    if not isinstance(t, str):
        return ""
    return t if len(t) <= max_len else (t[: max_len - 1] + "…")


def build_user_subgraph(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    *,
    user_original_id: int,
    max_user_movies: int = 10,
    include_2hop: bool = False,
    max_other_users: int = 10,
    max_other_user_movies: int = 3,
) -> Tuple[nx.Graph, List[str], List[str]]:
    G = nx.Graph()
    user_node = f"U{user_original_id}"
    G.add_node(user_node, kind="user")

    u_df = ratings[ratings["userId"] == user_original_id].sort_values("rating", ascending=False)
    u_df = u_df.head(max_user_movies)

    movie_nodes = []
    for _, row in u_df.iterrows():
        mid_internal = int(row["movie_id"])
        title = movies.iloc[mid_internal]["title"]
        m_node = f"M{mid_internal}"
        G.add_node(m_node, kind="movie", title=title)
        G.add_edge(user_node, m_node, rating=float(row["rating"]))
        movie_nodes.append(m_node)

    if include_2hop and movie_nodes:
        movie_internal_ids = [int(n[1:]) for n in movie_nodes]
        other = ratings[ratings["movie_id"].isin(movie_internal_ids)]
        other = other[other["userId"] != user_original_id]

        other_users = other["userId"].value_counts().head(max_other_users).index.tolist()

        for ou in other_users:
            u_node = f"U{ou}"
            G.add_node(u_node, kind="user")

            ou_df = ratings[(ratings["userId"] == ou) & (ratings["movie_id"].isin(movie_internal_ids))]
            for _, r2 in ou_df.iterrows():
                m_node = f"M{int(r2['movie_id'])}"
                G.add_edge(u_node, m_node, rating=float(r2["rating"]))

            ou_top = ratings[ratings["userId"] == ou].sort_values("rating", ascending=False).head(max_other_user_movies)
            for _, r3 in ou_top.iterrows():
                mid = int(r3["movie_id"])
                m_node = f"M{mid}"
                if not G.has_node(m_node):
                    title = movies.iloc[mid]["title"]
                    G.add_node(m_node, kind="movie", title=title)
                G.add_edge(u_node, m_node, rating=float(r3["rating"]))

    user_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "user"]
    movie_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "movie"]
    return G, user_nodes, movie_nodes


def draw_graph(G: nx.Graph, user_nodes: List[str], movie_nodes: List[str], title: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(title)
    pos = nx.spring_layout(G, seed=42, k=0.8)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.35, width=1.0)
    nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, node_shape="o", node_size=700, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=movie_nodes, node_shape="s", node_size=550, ax=ax)

    labels = {}
    for n in user_nodes:
        labels[n] = n
    for n in movie_nodes:
        labels[n] = _short_title(G.nodes[n].get("title", n))

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
    ax.axis("off")
    st.pyplot(fig)


# ===========================================================
# Header
# ===========================================================
st.title("🎥 LightGCN Movie Recommender")
st.caption("A recruiter-friendly demo of a Graph Neural Network (GNN) recommender on MovieLens data.")

with st.sidebar:
    st.subheader("Controls")
    beginner_mode = st.toggle("Beginner / Recruiter mode (more explanation)", value=True)
    tmdb_enabled = st.toggle("Show posters (TMDB)", value=bool(TMDB_API_KEY))
    st.divider()
    st.write("Tip: Train a checkpoint first for best recommendations.")


with st.spinner("Loading model + data..."):
    data_bundle = get_data_bundle()

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
    user_embs,
    item_embs,
    history,
    k_eval,
) = data_bundle

n_nodes = num_users + num_items
deg = compute_degrees(edge_index, n_nodes)

tab0, tab1, tab2, tab3 = st.tabs(
    ["🧭 Start Here (GNN Explained)", "🎯 Recommendations", "⚙️ Train Model", "🧠 Insights"]
)

with tab0:
    st.subheader("What you’re looking at")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users", f"{num_users:,}")
    c2.metric("Movies", f"{num_items:,}")
    c3.metric("Ratings", f"{len(ratings):,}")
    c4.metric("Graph edges (bi-dir)", f"{edge_index.shape[1]:,}")

    if beginner_mode:
        st.markdown(
            """
            **LightGCN is a Graph Neural Network for recommendations.**
            We treat the data as a **graph** where users and movies are nodes,
            and ratings create edges between them.
            """
        )

    st.markdown("### 1) From CSV rows ➜ a GNN graph")

    sample_row = ratings.sample(1, random_state=7).iloc[0]
    u_orig = int(sample_row["userId"])
    m_orig = int(sample_row["movieId"])
    u_int = int(sample_row["user_id"])
    m_int = int(sample_row["movie_id"])
    m_node_global = num_users + m_int

    left, right = st.columns([1, 1])
    with left:
        st.write("A single rating row:")
        st.dataframe(pd.DataFrame([{
            "userId (raw)": u_orig,
            "movieId (raw)": m_orig,
            "rating": float(sample_row["rating"]),
        }]), use_container_width=True, hide_index=True)

        st.write("We map raw IDs to contiguous indices:")
        st.code(
            f"userId {u_orig} -> user_id {u_int}\nmovieId {m_orig} -> movie_id {m_int}",
            language="text",
        )

    with right:
        st.write("Then we build a bipartite graph in one node ID space:")
        st.code(
            f"user nodes: 0 .. {num_users-1}\n"
            f"movie nodes: {num_users} .. {num_users+num_items-1}\n\n"
            f"edge: ({u_int}) <-> ({m_node_global})",
            language="text",
        )

    st.markdown("### 2) Visualize the user–movie graph (small subgraph)")
    user_pick = st.selectbox(
        "Pick a user (original userId)",
        options=sorted(ratings["userId"].unique().tolist()),
        index=0,
    )
    colA, colB, colC = st.columns([1, 1, 1])
    max_movies = colA.slider("Max movies (1-hop)", 5, 20, 10)
    include_2hop = colB.checkbox("Include 2-hop (adds other users)", value=False)
    max_other_users = colC.slider("Max other users (2-hop)", 3, 20, 10, disabled=not include_2hop)

    G, user_nodes, movie_nodes = build_user_subgraph(
        ratings,
        movies,
        user_original_id=int(user_pick),
        max_user_movies=max_movies,
        include_2hop=include_2hop,
        max_other_users=max_other_users,
    )
    draw_graph(G, user_nodes, movie_nodes, title="User–Movie Bipartite Subgraph")

    st.markdown("### 3) How message passing works (the GNN part)")
    if beginner_mode:
        st.latex(r"e^{(k+1)} = D^{-1/2} A D^{-1/2} e^{(k)}")

    u_internal = user2id[int(user_pick)]
    u_deg = deg[u_internal]

    u_movies = ratings[ratings["userId"] == int(user_pick)].sort_values("rating", ascending=False).head(8)
    rows = []
    for _, r in u_movies.iterrows():
        mid = int(r["movie_id"])
        mid_global = num_users + mid
        w = 0.0
        if u_deg > 0 and deg[mid_global] > 0:
            w = 1.0 / np.sqrt(u_deg * deg[mid_global])

        rows.append({
            "Movie": movies.iloc[mid]["title"],
            "Rating": float(r["rating"]),
            "deg(user)": int(u_deg),
            "deg(movie)": int(deg[mid_global]),
            "norm weight (1/sqrt(deg_u * deg_i))": float(w),
        })

    st.write("Example weights used when aggregating neighbors for this user:")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### 4) Tag cleaning and tag quality")
    m_internal_pick = st.selectbox(
        "Pick a movie (internal movie_id)",
        options=list(range(min(num_items, 500))),
        index=0,
    )
    m_row = movies.iloc[int(m_internal_pick)]

    t1, t2 = st.columns(2)
    with t1:
        st.write("Raw tags (noisy)")
        st.code(str(m_row.get("tag_raw", "No tags available")), language="text")
    with t2:
        st.write("Cleaned tags (reduced + filtered)")
        st.code(str(m_row.get("tag", "No tags available")), language="text")

    suggested = suggest_tags_from_neighbors(int(m_internal_pick), movies, item_embs, k_neighbors=25, top_tags=8)
    st.write("Suggested tags (from embedding neighbors):")
    st.write(", ".join(suggested) if suggested else "No suggestions available yet (train the model for better results).")


with tab1:
    st.subheader("Get recommendations for a user")

    col1, col2, col3 = st.columns([1, 1, 1])
    user_id = col1.selectbox(
        "User (original userId)",
        options=sorted(ratings["userId"].unique().tolist()),
        index=0,
    )
    top_k = col2.slider("Top-K recommendations", 5, 30, 10)
    alpha = col3.slider("Hybrid weight (tags)", 0.0, 0.6, 0.30, 0.05)

    if st.button("Generate recommendations", type="primary"):
        high, low, recs = generate_recommendations(int(user_id), data_bundle, top_k=top_k, alpha=alpha)

        st.markdown("### ⭐ Movies you rated highly")
        st.dataframe(high, use_container_width=True, hide_index=True)

        st.markdown("### 😐 Movies you rated low")
        st.dataframe(low, use_container_width=True, hide_index=True)

        st.markdown("### 🎯 Recommended movies (with score breakdown)")
        if recs.empty:
            st.warning("No recommendations for this user.")
        else:
            cols = st.columns(5)
            for idx, row in recs.iterrows():
                title = row.get("title", "Unknown")
                genres = row.get("genres", "")
                tag = row.get("tag", "")

                score_gnn = float(row.get("score_gnn", np.nan))
                score_tag = float(row.get("score_tag", np.nan))
                score_final = float(row.get("score_final", np.nan))

                year = None
                if "(" in title and title.rstrip().endswith(")"):
                    try:
                        year = int(title.split("(")[-1].replace(")", ""))
                    except Exception:
                        year = None

                poster_url = DEFAULT_POSTER
                if tmdb_enabled:
                    tmdb_movie = tmdb_search_movie(title, year)
                    poster_url = tmdb_get_poster(tmdb_movie)

                col = cols[idx % 5]
                with col:
                    st.image(poster_url, use_container_width=True)
                    st.markdown(f"**{title}**")
                    if genres:
                        st.caption(genres)

                    if beginner_mode:
                        st.write(
                            f"Graph score: `{score_gnn:.3f}`\n\n"
                            f"Tag score: `{score_tag:.3f}`\n\n"
                            f"Final: `{score_final:.3f}`"
                        )
                    if tag:
                        st.caption(f"Tags: {tag}")


with tab2:
    st.subheader("Train LightGCN (creates/updates `lightgcn_checkpoint.pth`)")

    c1, c2, c3, c4 = st.columns(4)
    embed_dim = c1.selectbox("Embedding dim", [32, 64, 128, 256], index=2)
    num_layers = c2.selectbox("Num layers", [1, 2, 3, 4], index=2)
    epochs = c3.slider("Epochs", 5, 60, 30)
    batch_size = c4.selectbox("Batch size", [1024, 2048, 4096, 8192], index=2)

    c5, c6, c7 = st.columns(3)
    lr = c5.number_input("Learning rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f")
    reg_lambda = c6.number_input("L2 regularization", min_value=0.0, max_value=1e-2, value=1e-4, format="%.6f")
    eval_every = c7.selectbox("Eval every N epochs", [1, 2, 5, 10], index=2)

    if st.button("Start training", type="primary"):
        st.session_state["train_log"] = []
        status = st.empty()
        chart = st.empty()
        table = st.empty()

        def cb(payload: dict):
            st.session_state["train_log"].append(payload)
            df = pd.DataFrame(st.session_state["train_log"])

            hit_txt = "—" if payload.get("hit@k") is None else f"{payload['hit@k'] * 100:.2f}%"
            status.info(f"Epoch {payload['epoch']} | loss {payload['loss']:.4f} | hit@10: {hit_txt}")

            chart.line_chart(df.set_index("epoch")[["loss"]], use_container_width=True)

            metric_cols = [c for c in ["hit@k", "recall@k", "ndcg@k", "sec"] if c in df.columns]
            table.dataframe(df[["epoch"] + metric_cols], use_container_width=True, hide_index=True)

        model = LightGCN(num_users, num_items, embed_dim=int(embed_dim), num_layers=int(num_layers))

        train_lightgcn(
            model,
            ratings,
            edge_index,
            num_users,
            num_items,
            epochs=int(epochs),
            lr=float(lr),
            batch_size=int(batch_size),
            reg_lambda=float(reg_lambda),
            eval_every=int(eval_every),
            eval_sample_size=300,
            k_eval=10,
            streamlit_callback=cb,
        )

        st.success("✅ Training complete. Reload the app to use the new checkpoint.")


with tab3:
    st.subheader("Model insights (embeddings + tags)")

    st.markdown("### 1) Embedding map (PCA over learned movie embeddings)")
    sample_n = st.slider("Movies to plot", 300, 2000, 800, 100)
    sample_idx = np.random.RandomState(7).choice(np.arange(num_items), size=min(sample_n, num_items), replace=False)

    emb = item_embs[sample_idx].numpy()
    coords = PCA(n_components=2, random_state=7).fit_transform(emb)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Movie embeddings (2D PCA)")
    ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig)

    st.markdown("### 2) Top cleaned tags (frequency)")
    all_tags = []
    for t in movies["tag"].astype(str).tolist():
        if t.lower().strip() == "no tags available":
            continue
        all_tags.extend(t.split())

    if all_tags:
        counts = pd.Series(all_tags).value_counts().head(25)
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        ax2.bar(counts.index.tolist(), counts.values.tolist())
        ax2.set_title("Top 25 tags (cleaned)")
        ax2.tick_params(axis="x", rotation=45)
        st.pyplot(fig2)
    else:
        st.info("No tags found.")