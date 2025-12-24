import pandas as pd

def leave_last_n_out(df: pd.DataFrame, n: int = 5):
    """
    For each user, sort ratings by timestamp and hold out the last n interactions as test.
    Returns: train_df, test_df
    """
    df = df.sort_values(["user_id", "timestamp"])

    train_parts = []
    test_parts = []

    for user_id, group in df.groupby("user_id"):
        if len(group) <= n:
            # If user doesn't have enough ratings, keep all in train
            train_parts.append(group)
        else:
            train_parts.append(group.iloc[:-n])
            test_parts.append(group.iloc[-n:])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True) if test_parts else df.iloc[0:0].copy()

    return train_df, test_df
