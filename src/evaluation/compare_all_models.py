import random
import numpy as np
import pandas as pd

from src.data.split import leave_last_n_out
from src.evaluation.metrics import precision_at_k, recall_at_k

from src.models.mf_sgd import MatrixFactorizationSGD
from src.models.bpr_mf import BPRMatrixFactorization


def evaluate_popularity(train_df, test_df, k=10, like_threshold=4.0, min_ratings=50, n_negatives=200):
    # Build global popularity ranking from TRAIN ONLY (avoid leakage)
    stats = train_df.groupby("movie_id").agg(
        num_ratings=("rating", "count"),
        avg_rating=("rating", "mean")
    )
    stats = stats[stats["num_ratings"] >= min_ratings]
    stats = stats.sort_values(["avg_rating", "num_ratings"], ascending=[False, False])
    popularity = stats.index.tolist()

    all_movie_ids = set(train_df["movie_id"].unique())
    train_by_user = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    test_likes = (
        test_df[test_df["rating"] >= like_threshold]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )

    precisions, recalls = [], []

    for u, relevant in test_likes.items():
        seen = train_by_user.get(u, set())
        unseen = list(all_movie_ids - seen - relevant)
        if not unseen:
            continue

        sampled_negatives = random.sample(unseen, min(n_negatives, len(unseen)))
        candidates = set(relevant) | set(sampled_negatives)

        # Recommend top-K from popularity list restricted to candidates
        recs = []
        for mid in popularity:
            if mid in candidates:
                recs.append(mid)
            if len(recs) == k:
                break

        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))

    return len(precisions), (sum(precisions) / len(precisions)), (sum(recalls) / len(recalls))


def evaluate_explicit_mf(train_df, test_df, k=10, like_threshold=4.0, n_negatives=200):
    # Train explicit MF (RMSE-style) on ratings
    model = MatrixFactorizationSGD()  # uses defaults from mf_sgd.py
    model.fit(train_df)

    all_movie_ids = set(model.idx_to_item)
    train_by_user = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    test_likes = (
        test_df[test_df["rating"] >= like_threshold]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )

    precisions, recalls = [], []

    for u, relevant in test_likes.items():
        already = train_by_user.get(u, set())
        unseen = list(all_movie_ids - already - relevant)
        if not unseen:
            continue

        sampled_negatives = random.sample(unseen, min(n_negatives, len(unseen)))
        candidates = list(relevant) + sampled_negatives

        recs = model.recommend_for_user(
            user_id=u,
            candidate_movie_ids=candidates,
            already_rated_ids=set(),  # candidates already filtered
            top_k=k
        )
        recommended_ids = [mid for mid, _ in recs]

        precisions.append(precision_at_k(recommended_ids, relevant, k))
        recalls.append(recall_at_k(recommended_ids, relevant, k))

    return len(precisions), (sum(precisions) / len(precisions)), (sum(recalls) / len(recalls))


def evaluate_bpr(train_df, test_df, k=10, like_threshold=4.0, n_negatives=200):
    # Train implicit MF (BPR ranking)
    model = BPRMatrixFactorization()  # uses defaults from bpr_mf.py
    model.fit(train_df, like_threshold=like_threshold)

    all_movie_ids = set(model.idx_to_item)
    train_by_user = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    test_likes = (
        test_df[test_df["rating"] >= like_threshold]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )

    precisions, recalls = [], []

    for u, relevant in test_likes.items():
        already = train_by_user.get(u, set())
        unseen = list(all_movie_ids - already - relevant)
        if not unseen:
            continue

        sampled_negatives = random.sample(unseen, min(n_negatives, len(unseen)))
        candidates = list(relevant) + sampled_negatives

        recs = model.recommend_for_user(
            user_id=u,
            candidate_movie_ids=candidates,
            already_rated_ids=set(),
            top_k=k
        )
        recommended_ids = [mid for mid, _ in recs]

        precisions.append(precision_at_k(recommended_ids, relevant, k))
        recalls.append(recall_at_k(recommended_ids, relevant, k))

    return len(precisions), (sum(precisions) / len(precisions)), (sum(recalls) / len(recalls))


def print_table(rows):
    # Simple fixed-width table printer (no extra dependencies)
    headers = ["Model", "Users", "Precision@10", "Recall@10"]
    col_widths = [max(len(str(r[i])) for r in ([headers] + rows)) for i in range(len(headers))]

    def fmt_row(r):
        return " | ".join(str(r[i]).ljust(col_widths[i]) for i in range(len(headers)))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for r in rows:
        print(fmt_row(r))


def main():
    # Reproducibility for negative sampling
    random.seed(42)
    np.random.seed(42)

    df = pd.read_csv("data/processed/ratings_with_titles.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    train_df, test_df = leave_last_n_out(df, n=5)

    K = 10
    like_threshold = 4.0
    n_negatives = 200

    rows = []

    users, p, r = evaluate_popularity(train_df, test_df, k=K, like_threshold=like_threshold, min_ratings=50, n_negatives=n_negatives)
    rows.append(["Popularity", users, f"{p:.4f}", f"{r:.4f}"])

    users, p, r = evaluate_explicit_mf(train_df, test_df, k=K, like_threshold=like_threshold, n_negatives=n_negatives)
    rows.append(["MF (RMSE)", users, f"{p:.4f}", f"{r:.4f}"])

    users, p, r = evaluate_bpr(train_df, test_df, k=K, like_threshold=like_threshold, n_negatives=n_negatives)
    rows.append(["BPR (implicit)", users, f"{p:.4f}", f"{r:.4f}"])

    print_table(rows)


if __name__ == "__main__":
    main()
