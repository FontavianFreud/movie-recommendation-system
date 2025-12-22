import pandas as pd

def main():
    ratings = pd.read_csv(
        "data/raw/ml-100k/u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    movies = pd.read_csv(
        "data/raw/ml-100k/u.item",
        sep="|",
        encoding="latin-1",
        header=None
    )
    movies = movies[[0, 1, 2]]
    movies.columns = ["movie_id", "title", "release_date"]

    df = ratings.merge(movies, on="movie_id", how="left")

    # basic cleaning
    df = df.dropna(subset=["title"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    df.to_csv("data/processed/ratings_with_titles.csv", index=False)
    print("Saved to data/processed/ratings_with_titles.csv")

if __name__ == "__main__":
    main()
