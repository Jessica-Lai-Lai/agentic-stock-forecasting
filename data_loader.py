import pandas as pd


def load_stock_data(file_path: str) -> pd.DataFrame:
    """
    Load stock price and news data from a CSV file.

    Expected columns:
    - date
    - ticker
    - close
    - news_summary
    """

    df = pd.read_csv(file_path)

    required_columns = {"date", "ticker", "close", "news_summary"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = df.dropna(subset=["date", "close", "news_summary"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def split_train_test(
    df: pd.DataFrame,
    train_size: int = 8,
    test_size: int = 4
):
    """
    Split the dataset into training data and testing data.

    train_df:
    Historical data used as input for forecasting.

    test_df:
    Future actual values used for evaluation.
    """

    if len(df) < train_size + test_size:
        raise ValueError("Not enough data for the requested train/test split.")

    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:train_size + test_size].copy()

    return train_df, test_df


def main():
    #file_path = "data/sample_stock_data.csv"
    df = load_stock_data("data/real_stock_data.csv")

    #df = load_stock_data(file_path)

    print("Full dataset:")
    print(df)

    train_df, test_df = split_train_test(df, train_size=8, test_size=4)

    print("\nTraining data:")
    print(train_df)

    print("\nTesting data:")
    print(test_df)


if __name__ == "__main__":
    main()