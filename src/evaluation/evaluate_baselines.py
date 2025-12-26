import pandas as pd

from src.data.split import leave_last_n_out
from src.evaluation.metrics import precision_at_k, recall_at_k


def build_popularity_ranking(train_df, min_ratings=50):
    """
    Build a global ranking of movies using training data only.
    Returns a list of movie_id ordered from best -> worst.
    """
    stats = train_df.groupby("movie_id").agg(
        num_ratings=("rating", "count"),
        avg_rating=("rating", "mean")
    )

    stats = stats[stats["num_ratings"] >= min_ratings]
    stats = stats.sort_values(["avg_rating", "num_ratings"], ascending=[False, False])

    return stats.index.tolist()  # index is movie_id


def evaluate_popularity_baseline(train_df, test_df, k=10, like_threshold=4.0, min_ratings=50):
    """
    Evaluate popularity baseline using Precision@K and Recall@K.

    For each user:
      - recommend the top-K movies globally (excluding what they rated in train)
      - relevant items = movies they rated >= like_threshold in test
    """
    popularity_list = build_popularity_ranking(train_df, min_ratings=min_ratings)

    # movies already rated per user (to avoid recommending seen items)
    train_by_user = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    # relevant test likes per user
    test_likes = (
        test_df[test_df["rating"] >= like_threshold]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )

    precisions = []
    recalls = []

    for u, relevant in test_likes.items():
        seen = train_by_user.get(u, set())

        # Recommend top-K from popularity list excluding seen movies
        recs = []
        for mid in popularity_list:
            if mid in seen:
                continue
            recs.append(mid)
            if len(recs) == k:
                break

        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))

    avg_p = sum(precisions) / len(precisions) if precisions else 0.0
    avg_r = sum(recalls) / len(recalls) if recalls else 0.0

    return len(precisions), avg_p, avg_r


def main():
    df = pd.read_csv("data/processed/ratings_with_titles.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    train_df, test_df = leave_last_n_out(df, n=5)

    users_eval, p10, r10 = evaluate_popularity_baseline(
        train_df, test_df, k=10, like_threshold=4.0, min_ratings=50
    )

    print("Popularity baseline")
    print("Users evaluated:", users_eval)
    print(f"Precision@10: {p10:.4f}")
    print(f"Recall@10:    {r10:.4f}")


if __name__ == "__main__":
    main()
