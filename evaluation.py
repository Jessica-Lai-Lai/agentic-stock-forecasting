import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def load_forecast_output(file_path: str = "outputs/forecast_output.json") -> dict:
    """
    Load forecast result from forecast_agent.py output JSON file.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Forecast output file not found: {file_path}. "
            "Please run forecast_agent.py first."
        )

    with open(file_path, "r") as f:
        result = json.load(f)

    return result


def calculate_mape(actual_values, predicted_values) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    """

    actual = np.array(actual_values)
    predicted = np.array(predicted_values)

    return np.mean(np.abs((actual - predicted) / actual)) * 100


def calculate_rmse(actual_values, predicted_values) -> float:
    """
    Calculate Root Mean Square Error.
    """

    return mean_squared_error(actual_values, predicted_values) ** 0.5


def save_outputs(result: dict, test_df: pd.DataFrame):
    """
    Save forecast result, reasoning output, and evaluation summary.

    Output files:
    - outputs/reasoning_output.json
    - outputs/forecast_results.csv
    - outputs/evaluation_summary.md
    """

    os.makedirs("outputs", exist_ok=True)

    actual_values = test_df["close"].tolist()
    predicted_values = result["final_forecast"]

    if len(actual_values) != len(predicted_values):
        raise ValueError("Actual values and predicted values must have the same length.")

    mape = calculate_mape(actual_values, predicted_values)
    rmse = calculate_rmse(actual_values, predicted_values)

    reasoning_path = "outputs/reasoning_output.json"
    forecast_path = "outputs/forecast_results.csv"
    summary_path = "outputs/evaluation_summary.md"

    with open(reasoning_path, "w") as f:
        json.dump(result, f, indent=2)

    forecast_df = pd.DataFrame({
        "date": test_df["date"].dt.strftime("%Y-%m-%d").tolist(),
        "ticker": test_df["ticker"].tolist(),
        "actual_close": actual_values,
        "predicted_close": predicted_values,
        "absolute_error": [
            abs(actual - predicted)
            for actual, predicted in zip(actual_values, predicted_values)
        ],
        "percentage_error": [
            abs((actual - predicted) / actual) * 100
            for actual, predicted in zip(actual_values, predicted_values)
        ]
    })

    forecast_df.to_csv(forecast_path, index=False)

    with open(summary_path, "w") as f:
        f.write("# Evaluation Summary\n\n")
        f.write("This file evaluates the LLM-generated forecast against actual stock prices.\n\n")
        f.write(f"MAPE: {mape:.2f}%\n\n")
        f.write(f"RMSE: {rmse:.2f}\n\n")
        f.write("## Notes\n\n")
        f.write("- MAPE measures average percentage error.\n")
        f.write("- RMSE measures the average size of prediction errors.\n")
        f.write("- This is a lightweight demo, not financial advice.\n")

    return {
        "mape": mape,
        "rmse": rmse,
        "forecast_path": forecast_path,
        "reasoning_path": reasoning_path,
        "summary_path": summary_path
    }


def main():
    from data_loader import load_stock_data, split_train_test

    #df = load_stock_data("data/sample_stock_data.csv")
    df = load_stock_data("data/real_stock_data.csv")
    train_df, test_df = split_train_test(df, train_size=8, test_size=4)

    forecast_result = load_forecast_output("outputs/forecast_output.json")

    evaluation_result = save_outputs(forecast_result, test_df)

    print("Evaluation completed.")
    print(f"MAPE: {evaluation_result['mape']:.2f}%")
    print(f"RMSE: {evaluation_result['rmse']:.2f}")
    print(f"Forecast CSV saved to: {evaluation_result['forecast_path']}")
    print(f"Reasoning JSON saved to: {evaluation_result['reasoning_path']}")
    print(f"Summary saved to: {evaluation_result['summary_path']}")


if __name__ == "__main__":
    main()