import json

from data_loader import load_stock_data, split_train_test
from context_agent import build_full_context
from forecast_agent import generate_forecast, save_forecast_output
from evaluation import save_outputs


def main():
    """
    Run the full agentic stock forecasting pipeline.

    Steps:
    1. Load stock price and news data
    2. Split data into train/test
    3. Build historical context
    4. Generate LLM forecast
    5. Save forecast output
    6. Evaluate forecast against actual values
    7. Save evaluation outputs
    """

    print("Step 1: Loading stock data...")
    #df = load_stock_data("data/sample_stock_data.csv")
    df = load_stock_data("data/real_stock_data.csv")

    print("Step 2: Splitting train/test data...")
    train_df, test_df = split_train_test(df, train_size=8, test_size=4)

    print("Step 3: Building historical context...")
    history_context = build_full_context(train_df)

    print("Step 4: Generating forecast with OpenAI API...")
    forecast_result = generate_forecast(history_context, horizon=4)

    print("Step 5: Saving forecast output...")
    forecast_output_path = save_forecast_output(
        forecast_result,
        output_path="outputs/forecast_output.json"
    )

    print("Step 6: Evaluating forecast...")
    evaluation_result = save_outputs(forecast_result, test_df)

    print("\nPipeline completed successfully.")
    print("--------------------------------")
    print(f"Forecast output JSON: {forecast_output_path}")
    print(f"Reasoning output JSON: {evaluation_result['reasoning_path']}")
    print(f"Forecast results CSV: {evaluation_result['forecast_path']}")
    print(f"Evaluation summary: {evaluation_result['summary_path']}")
    print("--------------------------------")
    print(f"MAPE: {evaluation_result['mape']:.2f}%")
    print(f"RMSE: {evaluation_result['rmse']:.2f}")

    print("\nForecast result preview:")
    print(json.dumps(forecast_result, indent=2))


if __name__ == "__main__":
    main()