import random
import pandas as pd

from src.data.split import leave_last_n_out
from src.models.bpr_mf import BPRMatrixFactorization
from src.evaluation.metrics import precision_at_k, recall_at_k


def main():

    random.seed(42)
    np.random.seed(42)


    df = pd.read_csv("data/processed/ratings_with_titles.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    train_df, test_df = leave_last_n_out(df, n=5)

    like_threshold = 4.0
    model = BPRMatrixFactorization(
        k=64,
        lr=0.05,
        reg=0.01,
        n_epochs=10,
        n_samples_per_epoch=150_000,  # adjust if too slow/fast
        seed=42,
    )
    model.fit(train_df, like_threshold=like_threshold)

    # Movies known to the model
    all_movie_ids = set(model.idx_to_item)

    # Already-rated items in train (for exclusion)
    train_by_user = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    # Relevant items in test: liked movies
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

        unseen = list(all_movie_ids - already - relevant)
        if len(unseen) == 0:
            continue

        sampled_negatives = random.sample(unseen, min(n_negatives, len(unseen)))
        candidates = list(relevant) + sampled_negatives

        recs = model.recommend_for_user(
            user_id=u,
            candidate_movie_ids=candidates,
            already_rated_ids=set(),  # candidates already filtered
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
