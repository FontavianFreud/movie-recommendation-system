import numpy as np
import pandas as pd


class MatrixFactorizationSGD:
    """
    Matrix Factorization with biases trained via SGD.

    Predict:
        r_hat(u,i) = global_mean + b_u + b_i + dot(P[u], Q[i])

    Where:
        P[u] is a k-dim user embedding
        Q[i] is a k-dim item embedding
        b_u, b_i are learned biases
    """

    def __init__(self, k=32, lr=0.01, reg=0.05, n_epochs=10, seed=42):
        self.k = k
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.seed = seed

        # Learned parameters (initialized in fit)
        self.global_mean = None
        self.b_u = None
        self.b_i = None
        self.P = None
        self.Q = None

        # Mappings from raw ids -> indices
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = []
        self.idx_to_item = []

    def _build_mappings(self, df: pd.DataFrame):
        """Map user_id and movie_id to contiguous indices 0..n-1."""
        users = df["user_id"].unique()
        items = df["movie_id"].unique()

        self.idx_to_user = users.tolist()
        self.idx_to_item = items.tolist()

        self.user_to_idx = {u: idx for idx, u in enumerate(self.idx_to_user)}
        self.item_to_idx = {i: idx for idx, i in enumerate(self.idx_to_item)}

    def fit(self, train_df: pd.DataFrame):
        """
        Train MF parameters using SGD on observed ratings.
        Expects columns: user_id, movie_id, rating
        """
        rng = np.random.default_rng(self.seed)

        self._build_mappings(train_df)

        n_users = len(self.idx_to_user)
        n_items = len(self.idx_to_item)

        # Initialize parameters
        self.global_mean = float(train_df["rating"].mean())
        self.b_u = np.zeros(n_users, dtype=np.float32)
        self.b_i = np.zeros(n_items, dtype=np.float32)

        # Small random initialization for embeddings
        self.P = rng.normal(0, 0.1, size=(n_users, self.k)).astype(np.float32)
        self.Q = rng.normal(0, 0.1, size=(n_items, self.k)).astype(np.float32)

        # Convert training data to index arrays for speed
        u_idx = train_df["user_id"].map(self.user_to_idx).to_numpy()
        i_idx = train_df["movie_id"].map(self.item_to_idx).to_numpy()
        ratings = train_df["rating"].to_numpy(dtype=np.float32)

        n = len(ratings)

        for epoch in range(1, self.n_epochs + 1):
            # Shuffle order each epoch (SGD stability)
            order = rng.permutation(n)

            se = 0.0  # sum squared error for monitoring

            for idx in order:
                u = u_idx[idx]
                i = i_idx[idx]
                r = ratings[idx]

                pred = self.global_mean + self.b_u[u] + self.b_i[i] + float(np.dot(self.P[u], self.Q[i]))
                err = r - pred

                se += float(err * err)

                # Bias updates (with L2 regularization)
                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])

                # Embedding updates (with L2 regularization)
                Pu = self.P[u].copy()
                Qi = self.Q[i].copy()

                self.P[u] += self.lr * (err * Qi - self.reg * Pu)
                self.Q[i] += self.lr * (err * Pu - self.reg * Qi)

            rmse = (se / n) ** 0.5
            print(f"Epoch {epoch}/{self.n_epochs} - Train RMSE: {rmse:.4f}")

        return self

    def predict_one(self, user_id: int, movie_id: int) -> float:
        """Predict rating for one user/movie. Falls back to global mean if unknown."""
        if user_id not in self.user_to_idx or movie_id not in self.item_to_idx:
            return float(self.global_mean)

        u = self.user_to_idx[user_id]
        i = self.item_to_idx[movie_id]

        pred = self.global_mean + self.b_u[u] + self.b_i[i] + float(np.dot(self.P[u], self.Q[i]))
        return float(pred)

    def recommend_for_user(self, user_id: int, candidate_movie_ids, already_rated_ids=None, top_k=10):
        """
        Rank candidate movies for a user using predicted rating.
        Returns list of (movie_id, score) sorted by score descending.
        """
        already_rated_ids = already_rated_ids or set()

        scored = []
        for mid in candidate_movie_ids:
            if mid in already_rated_ids:
                continue
            score = self.predict_one(user_id, mid)
            scored.append((mid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
