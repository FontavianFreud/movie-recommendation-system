import pandas as pd
from src.data.split import leave_last_n_out

def main():
    df = pd.read_csv("data/processed/ratings_with_titles.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    train_df, test_df = leave_last_n_out(df, n=5)

    print("Train size:", len(train_df))
    print("Test size:", len(test_df))
    print("Users:", df["user_id"].nunique())
    print("Example user counts (train/test):")

    u = df["user_id"].iloc[0]
    print("User", u, "train:", (train_df["user_id"] == u).sum(), "test:", (test_df["user_id"] == u).sum())

if __name__ == "__main__":
    main()
