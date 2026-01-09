import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data_preparation import prepare_data


# ============================================================
# LightGCN (Graph Neural Network for Collaborative Filtering)
# ============================================================
class LightGCN(nn.Module):
    """
    LightGCN learns user/item embeddings by propagating information over a
    user–item bipartite graph (no feature transforms / no nonlinearities).

    Core idea:
      e^{(k+1)} = D^{-1/2} A D^{-1/2} e^{(k)}
    Final embedding is the mean of embeddings from layer 0..K.
    """

    def __init__(self, num_users: int, num_items: int, embed_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        self.num_layers = num_layers
        self.cached_graph: Optional[torch.Tensor] = None

    def compute_graph(self, edge_index: torch.Tensor, num_users: int, num_items: int) -> torch.Tensor:
        """Precompute and cache normalized adjacency: D^-0.5 * A * D^-0.5."""
        if self.cached_graph is not None:
            return self.cached_graph

        n_nodes = num_users + num_items
        values = torch.ones(edge_index.shape[1], dtype=torch.float32, device=edge_index.device)

        adj = torch.sparse_coo_tensor(edge_index, values, (n_nodes, n_nodes)).coalesce()
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv = torch.pow(deg, -0.5)
        deg_inv[torch.isinf(deg_inv)] = 0.0

        idx = torch.arange(n_nodes, device=edge_index.device).unsqueeze(0).repeat(2, 1)
        deg_mat = torch.sparse_coo_tensor(idx, deg_inv, (n_nodes, n_nodes)).coalesce()

        dad = torch.sparse.mm(deg_mat, adj)
        dad = torch.sparse.mm(dad, deg_mat).coalesce()

        self.cached_graph = dad
        return dad

    def forward(self, edge_index: torch.Tensor, num_users: int, num_items: int) -> Tuple[torch.Tensor, torch.Tensor]:
        u0 = self.user_emb.weight
        i0 = self.item_emb.weight
        all_emb = torch.cat([u0, i0], dim=0)

        dad = self.compute_graph(edge_index, num_users, num_items)

        embs = [all_emb]
        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(dad, all_emb)
            embs.append(all_emb)

        final_emb = torch.stack(embs, dim=0).mean(dim=0)
        return final_emb[:num_users], final_emb[num_users:]


# ============================================================
# Evaluation helpers (sampled; leave-one-out style)
# ============================================================
def _build_user_pos(ratings, device: torch.device) -> Dict[int, torch.Tensor]:
    user_pos = {}
    for u in ratings["user_id"].unique():
        items = ratings.loc[ratings.user_id == u, "movie_id"].to_numpy()
        if len(items) > 0:
            user_pos[int(u)] = torch.tensor(items, dtype=torch.long, device=device)
    return user_pos


def _sample_eval_users(user_pos: Dict[int, torch.Tensor], sample_size: int, seed: int = 42) -> List[int]:
    rng = np.random.default_rng(seed)
    users = list(user_pos.keys())
    if not users:
        return []
    size = min(sample_size, len(users))
    return rng.choice(users, size=size, replace=False).tolist()


@torch.no_grad()
def eval_ranking_metrics(
    model: LightGCN,
    edge_index: torch.Tensor,
    num_users: int,
    num_items: int,
    user_pos: Dict[int, torch.Tensor],
    *,
    k: int = 10,
    eval_sample_size: int = 300,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Sampled leave-one-out evaluation:
    - for each sampled user, pick one positive item as the 'target'
    - rank all items, masking the user's other positives
    - compute Hit@K, Recall@K, NDCG@K
    """
    device = edge_index.device
    eval_users = _sample_eval_users(user_pos, eval_sample_size, seed=seed)
    if not eval_users:
        return {"hit@k": 0.0, "recall@k": 0.0, "ndcg@k": 0.0}

    user_emb, item_emb = model(edge_index, num_users, num_items)

    hits = 0
    recalls = []
    ndcgs = []

    for u in eval_users:
        pos = user_pos[u]
        if pos.numel() == 0:
            continue

        target = pos[torch.randint(0, pos.numel(), (1,), device=device)].item()
        scores = (user_emb[u] @ item_emb.T)

        if pos.numel() > 1:
            mask = pos[pos != target]
            scores[mask] = -float("inf")

        topk = torch.topk(scores, k).indices

        hit = int((topk == target).any().item())
        hits += hit
        recalls.append(float(hit))

        if hit:
            rank = (topk == target).nonzero(as_tuple=True)[0].item()
            ndcgs.append(1.0 / np.log2(rank + 2))
        else:
            ndcgs.append(0.0)

    n = max(1, len(eval_users))
    return {
        "hit@k": hits / n,
        "recall@k": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg@k": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


# ============================================================
# Training
# ============================================================
def train_lightgcn(
    model: LightGCN,
    ratings,
    edge_index: torch.Tensor,
    num_users: int,
    num_items: int,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 4096,
    reg_lambda: float = 1e-4,
    eval_every: int = 5,
    eval_sample_size: int = 300,
    k_eval: int = 10,
    seed: int = 42,
    streamlit_callback: Optional[Callable[[Dict], None]] = None,
) -> Dict[str, List[float]]:
    """
    BPR training for LightGCN with:
    - cached normalized adjacency
    - sampled leave-one-out evaluation (Hit/Recall/NDCG)
    - optional Streamlit callback for live dashboards
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    edge_index = edge_index.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    print(f"\n🚀 Training LightGCN on {device}")
    print(f"   epochs={epochs} | batch_size={batch_size} | lr={lr} | layers={model.num_layers} | dim={model.user_emb.embedding_dim}")
    print(f"   mixed_precision={'✓' if use_amp else '✗'}")

    user_pos = _build_user_pos(ratings, device)

    print("🔄 Caching normalized graph (D^-0.5 A D^-0.5) ...")
    with torch.no_grad():
        model.compute_graph(edge_index, num_users, num_items)
    print("✅ Graph cached.\n")

    num_batches = max(1, len(ratings) // batch_size)

    def bpr_loss(u_e, i_pos_e, i_neg_e):
        pos_scores = (u_e * i_pos_e).sum(dim=1)
        neg_scores = (u_e * i_neg_e).sum(dim=1)
        return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

    history = {
        "epoch": [],
        "loss": [],
        "hit@10": [],
        "recall@10": [],
        "ndcg@10": [],
        "sec": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        losses = []

        for _ in tqdm(range(num_batches), desc=f"Epoch {epoch}/{epochs}", leave=False):
            users = torch.randint(0, num_users, (batch_size,), device=device)

            pos_items = torch.empty(batch_size, dtype=torch.long, device=device)
            for idx, u in enumerate(users.tolist()):
                pos_list = user_pos.get(u)
                if pos_list is None or pos_list.numel() == 0:
                    pos_items[idx] = torch.randint(0, num_items, (1,), device=device).item()
                else:
                    pos_items[idx] = pos_list[torch.randint(0, pos_list.numel(), (1,), device=device)].item()

            neg_items = torch.randint(0, num_items, (batch_size,), device=device)

            if use_amp:
                with torch.amp.autocast("cuda", enabled=True):
                    user_emb, item_emb = model(edge_index, num_users, num_items)
                    u_e = user_emb[users]
                    i_pos_e = item_emb[pos_items]
                    i_neg_e = item_emb[neg_items]

                    loss = bpr_loss(u_e, i_pos_e, i_neg_e)
                    reg = reg_lambda * (u_e.norm(2).pow(2) + i_pos_e.norm(2).pow(2) + i_neg_e.norm(2).pow(2)) / (2 * batch_size)
                    total = loss + reg
                optimizer.zero_grad()
                scaler.scale(total).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                user_emb, item_emb = model(edge_index, num_users, num_items)
                u_e = user_emb[users]
                i_pos_e = item_emb[pos_items]
                i_neg_e = item_emb[neg_items]

                loss = bpr_loss(u_e, i_pos_e, i_neg_e)
                reg = reg_lambda * (u_e.norm(2).pow(2) + i_pos_e.norm(2).pow(2) + i_neg_e.norm(2).pow(2)) / (2 * batch_size)
                total = loss + reg

                optimizer.zero_grad()
                total.backward()
                optimizer.step()

            losses.append(float(total.detach().cpu().item()))

        avg_loss = float(np.mean(losses))
        sec = time.time() - t0

        hit = recall = ndcg = None
        if (epoch % eval_every == 0) or (epoch == epochs):
            model.eval()
            metrics = eval_ranking_metrics(
                model,
                edge_index,
                num_users,
                num_items,
                user_pos,
                k=k_eval,
                eval_sample_size=eval_sample_size,
                seed=seed + epoch,
            )
            hit = metrics["hit@k"]
            recall = metrics["recall@k"]
            ndcg = metrics["ndcg@k"]

        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        history["hit@10"].append(hit if hit is not None else float("nan"))
        history["recall@10"].append(recall if recall is not None else float("nan"))
        history["ndcg@10"].append(ndcg if ndcg is not None else float("nan"))
        history["sec"].append(sec)

        msg = f"Epoch {epoch:02d}/{epochs} | Loss {avg_loss:.4f} | {sec:.1f}s"
        if hit is not None:
            msg += f" | Hit@{k_eval} {hit*100:.2f}% | NDCG@{k_eval} {ndcg:.3f}"
        print(msg)

        if streamlit_callback:
            streamlit_callback({
                "epoch": epoch,
                "loss": avg_loss,
                "hit@k": hit,
                "recall@k": recall,
                "ndcg@k": ndcg,
                "sec": sec,
            })

    ckpt = {
        "model_state_dict": model.state_dict(),
        "num_users": num_users,
        "num_items": num_items,
        "embed_dim": model.user_emb.embedding_dim,
        "num_layers": model.num_layers,
        "history": history,
        "k_eval": k_eval,
    }
    torch.save(ckpt, "lightgcn_checkpoint.pth")
    print("\n✅ Saved checkpoint: lightgcn_checkpoint.pth\n")

    return history


if __name__ == "__main__":
    ratings, movies, num_users, num_items, edge_index, tag_features, movie_id_to_index, user2id, movie2id = prepare_data()
    model = LightGCN(num_users, num_items, embed_dim=128, num_layers=3)

    train_lightgcn(
        model,
        ratings,
        edge_index,
        num_users,
        num_items,
        epochs=30,
        batch_size=4096,
        eval_every=5,
        eval_sample_size=300,
        k_eval=10,
    )
