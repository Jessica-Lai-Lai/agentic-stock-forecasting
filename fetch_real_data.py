import os
import requests
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv


load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")


def fetch_weekly_stock_prices(
    ticker: str,
    start: str,
    end: str
) -> pd.DataFrame:
    """
    Fetch weekly stock closing prices from yfinance.

    Output:
    - date
    - ticker
    - close
    """

    stock = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1wk",
        auto_adjust=True,
        progress=False
    )

    if stock.empty:
        raise ValueError("No stock data downloaded. Please check ticker or date range.")

    stock = stock.reset_index()

    # yfinance may return MultiIndex columns, especially for newer versions.
    # This converts MultiIndex columns into simple column names.
    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = stock.columns.get_level_values(0)

    # Extract close price safely.
    close_values = stock["Close"]

    # Sometimes stock["Close"] is still a DataFrame with shape (n, 1).
    # Convert it into a 1D Series.
    if isinstance(close_values, pd.DataFrame):
        close_values = close_values.iloc[:, 0]

    df = pd.DataFrame()
    df["date"] = stock["Date"]
    df["ticker"] = ticker
    df["close"] = close_values.to_numpy().ravel()

    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = df.dropna(subset=["date", "close"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def fetch_finnhub_news_headlines(
    ticker: str,
    from_date: str,
    to_date: str,
    max_headlines: int = 3
) -> str:
    """
    Fetch company news headlines from Finnhub for a date range.
    Keep only headlines that are relevant to Nvidia / AI / semiconductor themes.
    """

    if not FINNHUB_API_KEY:
        raise ValueError("FINNHUB_API_KEY is missing. Please add it to your .env file.")

    url = "https://finnhub.io/api/v1/company-news"

    params = {
        "symbol": ticker,
        "from": from_date,
        "to": to_date,
        "token": FINNHUB_API_KEY
    }

    response = requests.get(url, params=params, timeout=20)

    if response.status_code != 200:
        raise ValueError(f"Finnhub API error: {response.status_code}, {response.text}")

    articles = response.json()

    if not articles:
        return (
            "No company-specific news found for this period; "
            "stock movement may reflect broader AI sector sentiment, semiconductor momentum, "
            "or macroeconomic conditions."
        )

    keywords = [
        "nvidia",
        "nvda",
        "gpu",
        "ai chip",
        "data center",
        "semiconductor",
        "earnings",
        "gtc",
        "blackwell",
        "cuda"
    ]

    headlines = []

    for article in articles:
        headline = article.get("headline", "")

        if not headline:
            continue

        headline_lower = headline.lower()

        if any(keyword in headline_lower for keyword in keywords):
            headlines.append(headline)

        if len(headlines) >= max_headlines:
            break

    if not headlines:
        return (
            "No highly relevant Nvidia-specific headline found for this period; "
            "stock movement may reflect broader market sentiment, AI sector momentum, "
            "semiconductor industry trends, or macroeconomic conditions."
        )

    return " | ".join(headlines)


def add_weekly_news_context(
    price_df: pd.DataFrame,
    ticker: str
) -> pd.DataFrame:
    """
    Add weekly Finnhub company news headlines to each weekly stock price row.
    """

    news_summaries = []

    for i in range(len(price_df)):
        week_start = price_df.loc[i, "date"]

        if i < len(price_df) - 1:
            week_end = price_df.loc[i + 1, "date"] - pd.Timedelta(days=1)
        else:
            week_end = week_start + pd.Timedelta(days=6)

        from_date = week_start.strftime("%Y-%m-%d")
        to_date = week_end.strftime("%Y-%m-%d")

        print(f"Fetching news for {ticker}: {from_date} to {to_date}")

        try:
            news_summary = fetch_finnhub_news_headlines(
                ticker=ticker,
                from_date=from_date,
                to_date=to_date,
                max_headlines=3
            )
        except Exception as error:
            news_summary = f"News fetch failed for this period: {error}"

        news_summaries.append(news_summary)

    price_df["news_summary"] = news_summaries

    return price_df


def main():
    ticker = "NVDA"
    start = "2026-03-01"
    end = "2026-06-01"


    print("Fetching weekly stock prices...")
    price_df = fetch_weekly_stock_prices(
        ticker=ticker,
        start=start,
        end=end
    )

    print("\nStock price data:")
    print(price_df)

    print("\nFetching weekly news context...")
    final_df = add_weekly_news_context(
        price_df=price_df,
        ticker=ticker
    )

    os.makedirs("data", exist_ok=True)

    output_path = "data/real_stock_data.csv"
    final_df.to_csv(output_path, index=False)

    print(f"\nSaved real stock data with news context to: {output_path}")
    print(final_df)


if __name__ == "__main__":
    main()