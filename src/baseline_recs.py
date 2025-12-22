import pandas as pd

def top_movies(df, min_ratings=50, n=10):
    stats = df.groupby("title").agg(
        num_ratings=("rating", "count"),
        avg_rating=("rating", "mean")
    )
    stats = stats[stats["num_ratings"] >= min_ratings]
    return stats.sort_values("avg_rating", ascending=False).head(n)

def main():
    df = pd.read_csv("data/processed/ratings_with_titles.csv")
    print(top_movies(df, min_ratings=100, n=15))

if __name__ == "__main__":
    main()
