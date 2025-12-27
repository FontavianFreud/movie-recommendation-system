import random
import pandas as pd

from src.data.split import leave_last_n_out
from src.models.mf_sgd import MatrixFactorizationSGD
from src.evaluation.metrics import precision_at_k, recall_at_k


def main():
    df = pd.read_csv("data/processed/ratings_with_titles.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    train_df, test_df = leave_last_n_out(df, n=5)

    model = MatrixFactorizationSGD(k=64, lr=0.01, reg=0.1, n_epochs=15, seed=42)
    model.fit(train_df)

    # Helper structures
    all_movie_ids = set(model.idx_to_item)
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
        already = train_by_user.get(u, set())

        # Pool of movies user has not seen
        unseen = list(all_movie_ids - already - relevant)
        if len(unseen) == 0:
            continue

        # Sample negatives
        sampled_negatives = random.sample(
            unseen, min(n_negatives, len(unseen))
        )

        # Candidate set = positives + negatives
        candidates = list(relevant) + sampled_negatives

        recs = model.recommend_for_user(
            user_id=u,
            candidate_movie_ids=candidates,
            already_rated_ids=set(),
            top_k=K
        )

        recommended_ids = [mid for mid, _ in recs]

        precisions.append(precision_at_k(recommended_ids, relevant, K))
        recalls.append(recall_at_k(recommended_ids, relevant, K))

    avg_p = sum(precisions) / len(precisions) if precisions else 0.0
    avg_r = sum(recalls) / len(recalls) if recalls else 0.0

    print(f"Users evaluated: {len(precisions)}")
    print(f"Precision@{K}: {avg_p:.4f}")
    print(f"Recall@{K}:    {avg_r:.4f}")


if __name__ == "__main__":
    main()
