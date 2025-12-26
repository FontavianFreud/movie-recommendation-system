import pandas as pd

from src.data.split import leave_last_n_out
from src.models.mf_sgd import MatrixFactorizationSGD
from src.evaluation.metrics import precision_at_k, recall_at_k


def main():
    # 1) Load processed data
    df = pd.read_csv("data/processed/ratings_with_titles.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 2) Split (time-aware per user)
    train_df, test_df = leave_last_n_out(df, n=5)

    # 3) Train MF model
    model = MatrixFactorizationSGD(k=32, lr=0.01, reg=0.05, n_epochs=8, seed=42)
    model.fit(train_df)

    # 4) Build helper structures for evaluation
    # Movies known to the model (trained items)
    all_movie_ids = set(model.idx_to_item)

    # For excluding already-rated movies per user
    train_by_user = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    # Relevant test items per user = movies rated >= 4 in test
    like_threshold = 4.0
    test_likes = (
        test_df[test_df["rating"] >= like_threshold]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )

    # 5) Evaluate Precision@K and Recall@K over users with at least 1 relevant test item
    K = 10
    precisions = []
    recalls = []

    users = sorted(test_likes.keys())

    for u in users:
        relevant = test_likes[u]
        already = train_by_user.get(u, set())

        # Candidate movies: all known movies minus what user already rated in train
        candidates = all_movie_ids  # we will filter inside recommend_for_user

        recs = model.recommend_for_user(
            user_id=u,
            candidate_movie_ids=candidates,
            already_rated_ids=already,
            top_k=K
        )

        recommended_ids = [mid for mid, score in recs]

        precisions.append(precision_at_k(recommended_ids, relevant, K))
        recalls.append(recall_at_k(recommended_ids, relevant, K))

    avg_p = sum(precisions) / len(precisions) if precisions else 0.0
    avg_r = sum(recalls) / len(recalls) if recalls else 0.0

    print(f"Users evaluated: {len(precisions)}")
    print(f"Precision@{K}: {avg_p:.4f}")
    print(f"Recall@{K}:    {avg_r:.4f}")


if __name__ == "__main__":
    main()
