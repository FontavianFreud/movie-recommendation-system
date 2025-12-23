import pandas as pd


def build_user_likes(df, like_threshold=4.0):
    """
    Convert raw ratings into a 'likes' table:
    user_id -> set/list of movies they liked.
    """
    likes = df[df["rating"] >= like_threshold][["user_id", "title"]]
    return likes


def similar_movies(df, seed_title, like_threshold=4.0, min_common_users=5, n=10):
    likes = build_user_likes(df, like_threshold=like_threshold)

    # Users who liked the seed movie
    seed_users = set(likes[likes["title"] == seed_title]["user_id"].unique())
    if not seed_users:
        return None

    # For each other movie, count how many of these seed users liked it
    other_likes = likes[likes["user_id"].isin(seed_users)]
    counts = other_likes["title"].value_counts()

    # Remove the seed itself
    counts = counts.drop(labels=[seed_title], errors="ignore")

    # Filter out weak signals
    counts = counts[counts >= min_common_users]

    return counts.head(n)


def main():
    df = pd.read_csv("data/processed/ratings_with_titles.csv")

    seed = "Star Wars (1977)"  # change this to test others
    recs = similar_movies(df, seed_title=seed, like_threshold=4.0, min_common_users=10, n=15)

    if recs is None:
        print("No users found who liked that title. Check spelling/title.")
        return

    print(f"Movies similar to: {seed}")
    print(recs)


if __name__ == "__main__":
    main()
