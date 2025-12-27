import random
import pandas as pd

from src.data.split import leave_last_n_out
from src.evaluation.metrics import precision_at_k, recall_at_k


def build_popularity_ranking(train_df, min_ratings=50):
    stats = train_df.groupby("movie_id").agg(
        num_ratings=("rating", "count"),
        avg_rating=("rating", "mean")
    )

    stats = stats[stats["num_ratings"] >= min_ratings]
    stats = stats.sort_values(["avg_rating", "num_ratings"], ascending=[False, False])

    return stats.index.tolist()


def main():
    df = pd.read_csv("data/processed/ratings_with_titles.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    train_df, test_df = leave_last_n_out(df, n=5)

    popularity = build_popularity_ranking(train_df, min_ratings=50)
    all_movie_ids = set(train_df["movie_id"].unique())
    train_by_user = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    like_threshold = 4.0
    test_likes = (
        test_df[test_df["rating"] >= like_threshold]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )

    K = 10
    n_negatives = 200

    precisions = []
    recalls = []

    for u, relevant in test_likes.items():
        seen = train_by_user.get(u, set())
        unseen = list(all_movie_ids - seen - relevant)

        if len(unseen) == 0:
            continue

        sampled_negatives = random.sample(
            unseen, min(n_negatives, len(unseen))
        )

        candidates = set(relevant) | set(sampled_negatives)

        recs = []
        for mid in popularity:
            if mid in candidates:
                recs.append(mid)
            if len(recs) == K:
                break

        precisions.append(precision_at_k(recs, relevant, K))
        recalls.append(recall_at_k(recs, relevant, K))

    avg_p = sum(precisions) / len(precisions) if precisions else 0.0
    avg_r = sum(recalls) / len(recalls) if recalls else 0.0

    print(f"Users evaluated: {len(precisions)}")
    print(f"Precision@{K}: {avg_p:.4f}")
    print(f"Recall@{K}:    {avg_r:.4f}")


if __name__ == "__main__":
    main()

