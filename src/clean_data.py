import pandas as pd


def main():
    # 1) Load ratings
    ratings = pd.read_csv(
        "data/raw/ml-100k/u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    # 2) Load movies (MovieLens 100K uses | delimiter and latin-1 encoding)
    movies = pd.read_csv(
        "data/raw/ml-100k/u.item",
        sep="|",
        encoding="latin-1",
        header=None
    )

    # Keep only: movie_id, title, release_date
    movies = movies[[0, 1, 2]]
    movies.columns = ["movie_id", "title", "release_date"]

    # 3) Merge into one table
    df = ratings.merge(movies, on="movie_id", how="left")

    # 4) Basic cleaning
    # Drop ratings that couldn't be matched to a title
    df = df.dropna(subset=["title"]).copy()

    # Convert unix seconds -> datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    # Release date is often "01-Jan-1995" style; parse to datetime if possible
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year

    # Enforce clean types (helps avoid subtle bugs later)
    df["user_id"] = df["user_id"].astype(int)
    df["movie_id"] = df["movie_id"].astype(int)
    df["rating"] = df["rating"].astype(float)

    # 5) Save processed artifact
    out_path = "data/processed/ratings_with_titles.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()

