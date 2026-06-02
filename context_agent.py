import pandas as pd


def build_history_context(df: pd.DataFrame) -> str:
    """
    Convert stock price and news data into a structured historical context.

    Input:
    - df: training dataframe with columns:
      date, ticker, close, news_summary

    Output:
    - A formatted string that can be sent to an LLM forecasting agent.
    """

    context_lines = []

    for index, row in df.iterrows():
        date = row["date"].strftime("%Y-%m-%d")
        ticker = row["ticker"]
        close_price = row["close"]
        news_summary = row["news_summary"]

        line = (
            f"Step {index + 1}:\n"
            f"Date: {date}\n"
            f"Ticker: {ticker}\n"
            f"Close Price: {close_price}\n"
            f"News Context: {news_summary}\n"
        )

        context_lines.append(line)

    history_context = "\n".join(context_lines)

    return history_context


def build_simple_statistics_context(df: pd.DataFrame) -> str:
    """
    Build simple numerical summary for the stock price trend.

    This gives the LLM extra numerical context without using many tokens.
    """

    first_price = df["close"].iloc[0]
    last_price = df["close"].iloc[-1]
    min_price = df["close"].min()
    max_price = df["close"].max()
    avg_price = df["close"].mean()

    total_change = last_price - first_price
    percentage_change = (total_change / first_price) * 100

    statistics_context = f"""
Basic Time Series Statistics:
- First close price: {first_price:.2f}
- Last close price: {last_price:.2f}
- Minimum close price: {min_price:.2f}
- Maximum close price: {max_price:.2f}
- Average close price: {avg_price:.2f}
- Total change: {total_change:.2f}
- Percentage change: {percentage_change:.2f}%
"""

    return statistics_context


def build_full_context(df: pd.DataFrame) -> str:
    """
    Combine structured historical timeline and simple statistics.
    """

    history_context = build_history_context(df)
    statistics_context = build_simple_statistics_context(df)

    full_context = f"""
{statistics_context}

Historical Timeline:
{history_context}
"""

    return full_context


def main():
    from data_loader import load_stock_data, split_train_test

    #df = load_stock_data("data/sample_stock_data.csv")
    df = load_stock_data("data/real_stock_data.csv")
    train_df, test_df = split_train_test(df, train_size=8, test_size=4)

    full_context = build_full_context(train_df)

    print(full_context)


if __name__ == "__main__":
    main()