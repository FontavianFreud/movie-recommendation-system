import numpy as np
import pandas as pd


class BPRMatrixFactorization:
    """
    Implicit-feedback Matrix Factorization using pairwise ranking (BPR-like).

    We do NOT predict ratings.
    We learn embeddings so that for a given user u:
        score(u, positive_item) > score(u, negative_item)

    score(u, i) = b_i + dot(P[u], Q[i])
    (We typically do not need user bias for BPR; item bias is useful.)
    """

    def __init__(
        self,
        k=64,
        lr=0.05,
        reg=0.01,
        n_epochs=10,
        n_samples_per_epoch=200_000,
        seed=42,
    ):
        self.k = k
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.n_samples_per_epoch = n_samples_per_epoch
        self.seed = seed

        self.P = None
        self.Q = None
        self.b_i = None

        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = []
        self.idx_to_item = []

        # user_idx -> set(item_idx) that are positives (liked)
        self.user_pos = None

    def _build_mappings(self, df: pd.DataFrame):
        users = df["user_id"].unique()
        items = df["movie_id"].unique()

        self.idx_to_user = users.tolist()
        self.idx_to_item = items.tolist()

        self.user_to_idx = {u: idx for idx, u in enumerate(self.idx_to_user)}
        self.item_to_idx = {i: idx for idx, i in enumerate(self.idx_to_item)}

    def fit(self, train_df: pd.DataFrame, like_threshold=4.0):
        """
        Train using implicit positives defined as rating >= like_threshold.

        Expects train_df columns: user_id, movie_id, rating
        """
        rng = np.random.default_rng(self.seed)

        self._build_mappings(train_df)

        n_users = len(self.idx_to_user)
        n_items = len(self.idx_to_item)

        # Build positives per user (as item indices)
        likes_df = train_df[train_df["rating"] >= like_threshold][["user_id", "movie_id"]].copy()
        likes_df["u_idx"] = likes_df["user_id"].map(self.user_to_idx)
        likes_df["i_idx"] = likes_df["movie_id"].map(self.item_to_idx)

        # user_pos[u] = set of positive item indices
        user_pos = [set() for _ in range(n_users)]
        for u, i in zip(likes_df["u_idx"].to_numpy(), likes_df["i_idx"].to_numpy()):
            if u >= 0 and i >= 0:
                user_pos[int(u)].add(int(i))

        self.user_pos = user_pos

        # Initialize parameters
        self.P = rng.normal(0, 0.1, size=(n_users, self.k)).astype(np.float32)
        self.Q = rng.normal(0, 0.1, size=(n_items, self.k)).astype(np.float32)
        self.b_i = np.zeros(n_items, dtype=np.float32)

        # Precompute users with at least 1 positive
        users_with_pos = np.array([u for u in range(n_users) if len(self.user_pos[u]) > 0], dtype=np.int32)
        if len(users_with_pos) == 0:
            raise ValueError("No users have positive (liked) items in training data. Lower like_threshold?")

        for epoch in range(1, self.n_epochs + 1):
            # Sample training pairs (u, i_pos, i_neg)
            for _ in range(self.n_samples_per_epoch):
                u = int(rng.choice(users_with_pos))

                # Sample a positive item i from user's positives
                i = int(rng.choice(list(self.user_pos[u])))

                # Sample a negative item j not in user's positives
                j = int(rng.integers(0, n_items))
                while j in self.user_pos[u]:
                    j = int(rng.integers(0, n_items))

                # Compute x_ui and x_uj
                x_ui = self.b_i[i] + float(np.dot(self.P[u], self.Q[i]))
                x_uj = self.b_i[j] + float(np.dot(self.P[u], self.Q[j]))
                x_uij = x_ui - x_uj

                # Sigmoid(-x_uij) = 1 / (1 + exp(x_uij))
                # This is the gradient factor for BPR
                sigmoid = 1.0 / (1.0 + np.exp(x_uij))

                # Cache old vectors
                Pu = self.P[u].copy()
                Qi = self.Q[i].copy()
                Qj = self.Q[j].copy()

                # Update embeddings (SGD + L2 reg)
                self.P[u] += self.lr * (sigmoid * (Qi - Qj) - self.reg * Pu)
                self.Q[i] += self.lr * (sigmoid * Pu - self.reg * Qi)
                self.Q[j] += self.lr * (-sigmoid * Pu - self.reg * Qj)

                # Update item biases
                self.b_i[i] += self.lr * (sigmoid - self.reg * self.b_i[i])
                self.b_i[j] += self.lr * (-sigmoid - self.reg * self.b_i[j])

            print(f"Epoch {epoch}/{self.n_epochs} - done")

        return self

    def score_one(self, user_id: int, movie_id: int) -> float:
        """
        Score (not rating) for one user/movie. Higher = more recommended.
        If unknown, return 0.
        """
        if user_id not in self.user_to_idx or movie_id not in self.item_to_idx:
            return 0.0

        u = self.user_to_idx[user_id]
        i = self.item_to_idx[movie_id]

        return float(self.b_i[i] + float(np.dot(self.P[u], self.Q[i])))

    def recommend_for_user(self, user_id: int, candidate_movie_ids, already_rated_ids=None, top_k=10):
        already_rated_ids = already_rated_ids or set()

        scored = []
        for mid in candidate_movie_ids:
            if mid in already_rated_ids:
                continue
            scored.append((mid, self.score_one(user_id, mid)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
