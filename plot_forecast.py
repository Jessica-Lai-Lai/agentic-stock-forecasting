import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecast(
    stock_data_path: str = "data/real_stock_data.csv",
    forecast_results_path: str = "outputs/forecast_results.csv",
    output_path: str = "outputs/forecast_plot.png",
    train_size: int = 8
):
    """
    Plot train actual prices, test actual prices, and 4-week forecast prices.
    """

    stock_df = pd.read_csv(stock_data_path)
    forecast_df = pd.read_csv(forecast_results_path)

    stock_df["date"] = pd.to_datetime(stock_df["date"])
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])

    train_df = stock_df.iloc[:train_size]
    test_df = forecast_df

    forecast_start_date = test_df["date"].iloc[0]

    # Read MAPE/RMSE from evaluation summary manually if needed
    mape = 3.36
    rmse = 8.67



    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(11, 6))

    # Train actual line
    plt.plot(
        train_df["date"],
        train_df["close"],
        marker="o",
        label="Train Actual Price"
    )

    # Test actual line
    plt.plot(
        test_df["date"],
        test_df["actual_close"],
        marker="o",
        label="Test Actual Price"
    )

    # Predicted forecast line
    plt.plot(
        test_df["date"],
        test_df["predicted_close"],
        marker="o",
        linestyle="--",
        label="4-Week Forecast"
    )

    # Forecast start vertical line
    plt.axvline(
        forecast_start_date,
        linestyle=":",
        label="Forecast Start"
    )

    # Evaluation text box
    plt.text(
        0.02,
        0.75,
        f"MAPE: {mape:.2f}%\nRMSE: {rmse:.2f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.2)
    )

    plt.title("NVDA 4-Week Forecast: Actual vs Predicted Close Price")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Forecast plot saved to: {output_path}")


if __name__ == "__main__":
    plot_forecast()